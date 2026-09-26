/**
 * @file nlm.c
 * @brief Implementation of Non-Local Means denoising.
 *
 * The image is first copied into a buffer padded by (patch + search) pixels
 * on every side, mirroring it at the borders. All neighborhood accesses then
 * stay inside the buffer, so the inner loops are free of bounds checks.
 *
 * Two methods compute the same result:
 *
 * Direct: for every pixel and every candidate offset in the search window,
 * the two patches are compared pixel by pixel: up to (2S+1)^2 x (2P+1)^2
 * operations per pixel (11025 with the defaults). A comparison is abandoned
 * as soon as its partial sum passes the point where the weight becomes
 * negligible, which saves most of the work on images with little noise.
 *
 * Integral: the loops are exchanged. For every offset, the squared
 * differences between the image and its shifted copy are summed into an
 * integral image (summed-area table), from which the patch sum of any pixel
 * is read with 4 lookups, independently of the patch size: about
 * (2S+1)^2 x a small constant operations per pixel (Darbon et al.,
 * "Fast nonlocal filtering applied to electron cryomicroscopy", ISBI 2008).
 *
 * A weight depends only on the (integer) patch sum, so the weights of all
 * sums up to the cutoff are computed once per image into a table, instead
 * of calling expf() for every candidate; the table holds exactly the values
 * expf() would return.
 *
 * Both methods add the contributions of the offsets to every pixel in the
 * same order and with the same arithmetic, so their outputs are bit-for-bit
 * identical, and identical for any thread count and parallelization mode.
 */

#define _POSIX_C_SOURCE 200809L

#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "nlm.h"

#ifndef M_PI
#	define M_PI 3.14159265358979323846
#endif

/*
 * Weights with an exponent argument above this are below e^-30 (~1e-13)
 * and are skipped; they cannot change an 8-bit result.
 */
#define NLM_MAX_ARG 30.0

/*
 * Largest weight table, in entries (4 bytes each). The table covers every
 * patch sum up to the cutoff, which grows with sigma^2 and the patch area;
 * beyond this size (very strong noise with large patches) the weights are
 * computed with expf() instead, with identical results.
 */
#define NLM_MAX_TABLE (1L << 22)

/*
 * Target height of the bands processed by the integral method. A band's
 * integral image and weight accumulators (~ 16 bytes per pixel of band)
 * should stay in the per-core L2 cache while all offsets are processed.
 */
#define NLM_BAND_ROWS 32

/**
 * @struct NlmContext
 * @brief Everything the per-method loops need, computed once per image.
 */
typedef struct {
	const unsigned char *pad; /**< Mirrored, padded input */
	size_t pw;                /**< Width of the padded input */
	long w;                   /**< Image width */
	long h;                   /**< Image height */
	int p;                    /**< Patch radius */
	int s;                    /**< Search radius */
	double offset;            /**< 2 sigma^2 * patch area */
	double inv;               /**< 1 / (h^2 * patch area) */
	int cutoff;               /**< Largest patch sum with a non-negligible weight */
	const float *wtab;        /**< Weight per patch sum 0..cutoff, 0 at cutoff+1; or NULL */
} NlmContext;

/* ------------------------------------------------------------------------- */
/*                            Static Helper Functions                        */
/* ------------------------------------------------------------------------- */

/**
 * @brief Mirrors an index into [0, n) (symmetric: -1 -> 0, n -> n-1).
 *
 * Works for any distance outside the range, so images smaller than the
 * padding are handled too.
 */
static size_t
reflect(long i, long n)
{
	long period = 2 * n;

	i %= period;
	if (i < 0)
		i += period;
	if (i >= n)
		i = period - 1 - i;

	return (size_t)i;
}

/**
 * @brief Copies an image into a new buffer padded by @p r mirrored pixels.
 *
 * @param[in]  src      Source image.
 * @param[in]  r        Padding on every side.
 * @param[in]  parallel Non-zero to parallelize over rows.
 * @param[out] pw       Width of the padded buffer.
 *
 * @return Newly allocated buffer of (W + 2r) x (H + 2r) bytes, or NULL.
 */
static unsigned char*
pad_image(const Image *src, unsigned int r, int parallel, size_t *pw)
{
	const long W = (long)src->width;
	const long H = (long)src->height;
	const size_t PW = (size_t)W + 2 * r;
	const size_t PH = (size_t)H + 2 * r;
	unsigned char *p = malloc(PW * PH);

	if (!p)
		return NULL;

	#pragma omp parallel for schedule(static) if(parallel)
	for (long y = 0; y < (long)PH; y++) {
		const unsigned char *row = src->data + reflect(y - (long)r, H) * (size_t)W;
		unsigned char *dst = p + (size_t)y * PW;

		for (long x = 0; x < (long)PW; x++)
			dst[x] = row[reflect(x - (long)r, W)];
	}

	*pw = PW;
	return p;
}

/**
 * @brief Weight of a candidate with patch sum @p ssd (already <= cutoff).
 *
 * Shared by both methods so that they compute bit-identical weights.
 */
static inline float
weight(const NlmContext *c, int ssd)
{
	double arg = ((double)ssd - c->offset) * c->inv;
	return expf(-(float)(arg > 0.0 ? arg : 0.0));
}

/**
 * @brief Weight of a candidate with any patch sum, 0 beyond the cutoff.
 *
 * Reads the precomputed table when there is one. Sums beyond the cutoff
 * map to the table's last entry, which is 0 (adding a zero weight leaves
 * every accumulator unchanged).
 */
static inline float
lookup(const NlmContext *c, int ssd)
{
	if (c->wtab)
		return c->wtab[ssd > c->cutoff ? c->cutoff + 1 : ssd];

	return (ssd > c->cutoff) ? 0.0f : weight(c, ssd);
}

/**
 * @brief Builds the table of weights for every patch sum.
 *
 * Entry k holds weight(k) for k <= cutoff and 0 at cutoff + 1. The entries
 * come from the same weight() function the loops would otherwise call, so
 * using the table changes nothing in the output.
 *
 * @return Newly allocated table, or NULL if it would exceed NLM_MAX_TABLE
 *         entries or allocation fails (the loops then call weight()).
 */
static float*
build_table(const NlmContext *c, int parallel)
{
	long n = (long)c->cutoff + 2;
	float *t;

	if (c->cutoff >= INT_MAX - 1 || n > NLM_MAX_TABLE)
		return NULL;

	t = malloc((size_t)n * sizeof(float));
	if (!t)
		return NULL;

	#pragma omp parallel for schedule(static) if(parallel)
	for (long k = 0; k <= c->cutoff; k++)
		t[k] = weight(c, (int)k);

	t[c->cutoff + 1] = 0.0f;
	return t;
}

/**
 * @brief Final value of a pixel from its accumulated weights.
 *
 * The center pixel counts as much as its most similar neighbor.
 */
static inline unsigned char
finish(float wsum, float vsum, float wmax, unsigned char center)
{
	long value;

	if (wmax == 0.0f)
		wmax = 1.0f;
	wsum += wmax;
	vsum += wmax * (float)center;

	value = lroundf(vsum / wsum);
	return (unsigned char)(value < 0 ? 0 : value > 255 ? 255 : value);
}

/**
 * @brief Direct method: patches compared pixel by pixel.
 *
 * With @p parallel set, rows are distributed over the threads.
 */
static int
nlm_direct(const NlmContext *c, unsigned char *out, int parallel)
{
	const long W = c->w;
	const long H = c->h;
	const int P = c->p;
	const int S = c->s;
	const int side = 2 * P + 1;
	const long PW = (long)c->pw;

	#pragma omp parallel for schedule(static) if(parallel)
	for (long y = 0; y < H; y++) {
		for (long x = 0; x < W; x++) {
			const unsigned char *cp = c->pad + (size_t)(y + P + S) * c->pw + (size_t)(x + P + S);
			float wsum = 0.0f, vsum = 0.0f, wmax = 0.0f;

			for (int dy = -S; dy <= S; dy++) {
				for (int dx = -S; dx <= S; dx++) {
					const unsigned char *qp = cp + (long)dy * PW + dx;
					float w;
					int ssd = 0;

					if (dx == 0 && dy == 0)
						continue;

					/*
					 * Sum of squared differences of the two patches, one
					 * row at a time. The sum only grows, so once it passes
					 * the cutoff the rest of the patch is skipped; the
					 * candidate is dropped either way.
					 */
					for (int py = -P; py <= P && ssd <= c->cutoff; py++) {
						const unsigned char *a = cp + (long)py * PW - P;
						const unsigned char *b = qp + (long)py * PW - P;

						for (int px = 0; px < side; px++) {
							int d = (int)a[px] - (int)b[px];
							ssd += d * d;
						}
					}

					if (ssd > c->cutoff)
						continue;

					w = lookup(c, ssd);
					wsum += w;
					vsum += w * (float)*qp;
					if (w > wmax)
						wmax = w;
				}
			}

			out[(size_t)y * W + x] = finish(wsum, vsum, wmax, *cp);
		}
	}

	return IMG_OK;
}

/**
 * @brief Integral method on the rows [y0, y1) of the image.
 *
 * For every offset, the squared differences over the band (plus P rows and
 * columns of patch halo on each side) are summed into the integral image
 * @p ii, and every pixel's patch sum is read from it with 4 lookups.
 *
 * The integral image holds 32-bit unsigned values that are allowed to
 * wrap: a patch sum is the difference of 4 entries, which is exact modulo
 * 2^32, and a patch sum itself is below 2^32 (at most 441 * 255^2).
 *
 * @param[in]  c    Context.
 * @param[in]  y0   First row of the band.
 * @param[in]  y1   One past the last row of the band.
 * @param[in]  ii   Scratch: (y1-y0 + 2P + 1) x (W + 2P + 1) values.
 * @param[in]  acc  Scratch: 3 x (y1-y0) x W floats (weight sum, weighted
 *                  value sum, maximum weight).
 * @param[out] out  Output image (rows [y0, y1) are written).
 */
static void
integral_band(const NlmContext *c, long y0, long y1, uint32_t *ii, float *acc,
              unsigned char *out)
{
	const long W = c->w;
	const int P = c->p;
	const int S = c->s;
	const long R = P + S;
	const long PW = (long)c->pw;
	const long rows = y1 - y0;
	const long nv = rows + 2 * P;           /* rows of differences */
	const long nu = W + 2 * P;              /* columns of differences */
	const size_t iw = (size_t)nu + 1;       /* integral width (zero column) */
	const size_t n = (size_t)rows * (size_t)W;
	float *wsum = acc;
	float *vsum = acc + n;
	float *wmax = acc + 2 * n;

	memset(acc, 0, 3 * n * sizeof(float));
	memset(ii, 0, iw * sizeof(uint32_t)); /* zero row */

	for (int dy = -S; dy <= S; dy++) {
		for (int dx = -S; dx <= S; dx++) {
			if (dx == 0 && dy == 0)
				continue;

			/* integral image of the squared differences for this offset */
			for (long i = 0; i < nv; i++) {
				const unsigned char *a = c->pad + (size_t)(y0 + R - P + i) * c->pw + (size_t)(R - P);
				const unsigned char *b = a + (long)dy * PW + dx;
				const uint32_t *prev = ii + (size_t)i * iw;
				uint32_t *cur = ii + (size_t)(i + 1) * iw;
				uint32_t rs = 0;

				cur[0] = 0;
				for (long j = 0; j < nu; j++) {
					int d = (int)a[j] - (int)b[j];
					rs += (uint32_t)(d * d);
					cur[j + 1] = prev[j + 1] + rs;
				}
			}

			/* patch sums from 4 lookups, then weights */
			for (long y = 0; y < rows; y++) {
				const uint32_t *top = ii + (size_t)y * iw;
				const uint32_t *bot = ii + (size_t)(y + 2 * P + 1) * iw;
				const unsigned char *q = c->pad + (size_t)(y0 + y + R + dy) * c->pw + (size_t)(R + dx);
				float *ws = wsum + (size_t)y * W;
				float *vs = vsum + (size_t)y * W;
				float *wm = wmax + (size_t)y * W;

				/*
				 * Rejected candidates are skipped with a branch rather than
				 * added with weight 0: on clean images nearly all of them
				 * are rejected, the branch is predicted correctly, and the
				 * accumulators are not touched. A branch-free version is
				 * ~25% faster on very noisy images but ~50% slower on
				 * clean ones.
				 */
				for (long x = 0; x < W; x++) {
					int ssd = (int)(bot[x + 2 * P + 1] - top[x + 2 * P + 1] - bot[x] + top[x]);
					float w;

					if (ssd > c->cutoff)
						continue;

					w = lookup(c, ssd);
					ws[x] += w;
					vs[x] += w * (float)q[x];
					if (w > wm[x])
						wm[x] = w;
				}
			}
		}
	}

	for (long y = 0; y < rows; y++) {
		const unsigned char *center = c->pad + (size_t)(y0 + y + R) * c->pw + (size_t)R;
		unsigned char *o = out + (size_t)(y0 + y) * W;

		for (long x = 0; x < W; x++) {
			size_t k = (size_t)y * W + x;
			o[x] = finish(wsum[k], vsum[k], wmax[k], center[x]);
		}
	}
}

/**
 * @brief Integral method over the whole image.
 *
 * The image is split into bands of about NLM_BAND_ROWS rows. With
 * @p parallel set, the bands are distributed over the threads, each with
 * its own scratch buffers; the number of bands is rounded up to a multiple
 * of the thread count so that no thread is left with a partial last round.
 * Without it, the calling thread processes the bands in turn.
 *
 * @return IMG_OK, or IMG_ERR_NOMEM.
 */
static int
nlm_integral(const NlmContext *c, unsigned char *out, int parallel)
{
	const long H = c->h;
	const int P = c->p;
	long nb = (H + NLM_BAND_ROWS - 1) / NLM_BAND_ROWS;
	long band;
	int failed = 0;

	if (parallel) {
		long t = omp_get_max_threads();
		nb = ((nb + t - 1) / t) * t;
	}
	band = (H + nb - 1) / nb;
	nb = (H + band - 1) / band;

	#pragma omp parallel if(parallel)
	{
		const size_t iw = (size_t)c->w + 2 * P + 1;
		uint32_t *ii = malloc((size_t)(band + 2 * P + 1) * iw * sizeof(uint32_t));
		float *acc = malloc(3 * (size_t)band * (size_t)c->w * sizeof(float));

		if (!ii || !acc) {
			#pragma omp atomic write
			failed = 1;
		}

		#pragma omp for schedule(dynamic)
		for (long b = 0; b < nb; b++) {
			long y0 = b * band;
			long y1 = (y0 + band < H) ? y0 + band : H;

			if (ii && acc)
				integral_band(c, y0, y1, ii, acc, out);
		}

		free(ii);
		free(acc);
	}

	return failed ? IMG_ERR_NOMEM : IMG_OK;
}

/* ------------------------------------------------------------------------- */
/*                            Public API Functions                           */
/* ------------------------------------------------------------------------- */

/**
 * @brief Estimates the noise standard deviation of an image.
 *
 * Immerkaer's method: the image is convolved with a Laplacian-difference
 * mask that cancels smooth structure, and the mean absolute response is
 * scaled to a standard deviation. The sum is accumulated in integers, so the
 * estimate is exact and independent of the thread count. Strong texture is
 * partly counted as noise, so the estimate is an upper bound on such images.
 *
 * @param[in] img      Image (at least 3x3; smaller images return 0).
 * @param[in] parallel Non-zero to parallelize over rows.
 *
 * @return Estimated noise standard deviation in gray levels.
 */
double
nlm_noise_estimate(const Image *img, int parallel)
{
	const long W = (long)img->width;
	const long H = (long)img->height;
	unsigned long long sum = 0;

	if (W < 3 || H < 3)
		return 0.0;

	/*
	 * Mask:  1 -2  1
	 *       -2  4 -2
	 *        1 -2  1
	 */
	#pragma omp parallel for schedule(static) reduction(+:sum) if(parallel)
	for (long y = 1; y < H - 1; y++) {
		const unsigned char *a = img->data + (size_t)(y - 1) * W;
		const unsigned char *b = a + W;
		const unsigned char *c = b + W;
		unsigned long long row = 0;

		for (long x = 1; x < W - 1; x++) {
			int v = a[x - 1] - 2 * a[x] + a[x + 1]
			      - 2 * b[x - 1] + 4 * b[x] - 2 * b[x + 1]
			      + c[x - 1] - 2 * c[x] + c[x + 1];
			row += (unsigned long long)(v < 0 ? -v : v);
		}

		sum += row;
	}

	return sqrt(M_PI / 2.0) * (double)sum / (6.0 * (double)(W - 2) * (double)(H - 2));
}

/**
 * @brief Denoises an image with Non-Local Means.
 *
 * Allocates @p dst. With @p parallel set, the work of the image is
 * distributed over the OpenMP threads (rows for the direct method, bands
 * for the integral method); otherwise the image is processed by the
 * calling thread alone (used when parallelizing across images).
 *
 * @param[in]  src       Noisy image.
 * @param[out] dst       Denoised image (allocated, caller frees).
 * @param[in]  p         Filter parameters.
 * @param[in]  parallel  Non-zero to parallelize within the image.
 * @param[out] sigma_out Noise standard deviation used (may be NULL).
 *
 * @return IMG_OK, or IMG_ERR_NOMEM / IMG_ERR_SIZE.
 */
int
nlm_denoise(const Image *src, Image *dst, const NlmParams *p, int parallel,
            double *sigma_out)
{
	const long W = (long)src->width;
	const long H = (long)src->height;
	const double area = (double)(2 * p->patch + 1) * (double)(2 * p->patch + 1);
	unsigned char *pad;
	float *tab;
	NlmContext c;
	double sigma, h, cut;
	int ret;

	dst->width = dst->height = 0;
	dst->data = NULL;

	if (!src->data || W == 0 || H == 0)
		return IMG_ERR_SIZE;

	sigma = (p->sigma >= 0.0) ? p->sigma : nlm_noise_estimate(src, parallel);
	if (sigma_out)
		*sigma_out = sigma;

	dst->data = malloc((size_t)W * H);
	if (!dst->data)
		return IMG_ERR_NOMEM;
	dst->width = src->width;
	dst->height = src->height;

	/* no noise, nothing to remove */
	h = p->h_factor * sigma;
	if (h <= 0.0) {
		memcpy(dst->data, src->data, (size_t)W * H);
		return IMG_OK;
	}

	pad = pad_image(src, p->patch + p->search, parallel, &c.pw);
	if (!pad) {
		image_free(dst);
		return IMG_ERR_NOMEM;
	}

	c.pad = pad;
	c.w = W;
	c.h = H;
	c.p = (int)p->patch;
	c.s = (int)p->search;

	/*
	 * Work on patch sums instead of means: the exponent
	 *   (d^2 - 2 sigma^2) / h^2  with  d^2 = ssd / area
	 * equals (ssd - offset) * inv.
	 */
	c.offset = 2.0 * sigma * sigma * area;
	c.inv = 1.0 / (h * h * area);

	/*
	 * Largest patch sum whose weight is not negligible: arg <= NLM_MAX_ARG
	 * <=> ssd <= offset + NLM_MAX_ARG / inv. Kept as an integer so the
	 * inner loops compare integers; clamped to fit.
	 */
	cut = floor(c.offset + NLM_MAX_ARG / c.inv);
	c.cutoff = (cut >= (double)INT_MAX) ? INT_MAX : (int)cut;

	/* one expf() per possible patch sum instead of one per candidate */
	tab = build_table(&c, parallel);
	c.wtab = tab;

	if (p->method == NLM_DIRECT)
		ret = nlm_direct(&c, dst->data, parallel);
	else
		ret = nlm_integral(&c, dst->data, parallel);

	free(tab);
	free(pad);

	if (ret != IMG_OK)
		image_free(dst);

	return ret;
}

/**
 * @brief Returns the name of an NLM method ("direct", "integral").
 *
 * @param[in] method NLM_* value.
 *
 * @return Static name string.
 */
const char*
nlm_method_name(int method)
{
	switch (method) {
	case NLM_DIRECT:   return "direct";
	case NLM_INTEGRAL: return "integral";
	default:           return "unknown";
	}
}

/**
 * @brief Parses an NLM method name.
 *
 * @param[in] name Method name.
 *
 * @return NLM_* value, or -1 if unknown.
 */
int
nlm_method_from_name(const char *name)
{
	if (!name)
		return -1;
	if (strcmp(name, "direct") == 0)
		return NLM_DIRECT;
	if (strcmp(name, "integral") == 0)
		return NLM_INTEGRAL;
	return -1;
}
