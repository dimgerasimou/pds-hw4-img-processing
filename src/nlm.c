/**
 * @file nlm.c
 * @brief Implementation of Non-Local Means denoising.
 *
 * The image is first copied into a buffer padded by (patch + search) pixels
 * on every side, mirroring it at the borders, so that all neighborhood
 * accesses stay inside the buffer and the inner loops need no bounds checks.
 *
 * Patch distances are computed with integral images (Darbon et al.,
 * "Fast nonlocal filtering applied to electron cryomicroscopy", ISBI 2008):
 * for every offset of the search window, the squared differences between
 * the image and its shifted copy are summed into an integral image
 * (summed-area table), from which the patch sum of any pixel is read with 4
 * lookups. The cost is about (2S+1)^2 per pixel, independent of the patch
 * size, instead of (2S+1)^2 (2P+1)^2 when patches are compared directly.
 *
 * A weight depends only on the (integer) patch sum, so the weights of all
 * sums up to the cutoff are computed once per image into a table, instead
 * of calling expf() for every candidate; the table holds exactly the values
 * expf() would return.
 *
 * The work of an image is split into bands of rows, which can run on any
 * thread in any order: every output pixel depends only on the input, so the
 * result is identical for any thread count.
 */

#define _POSIX_C_SOURCE 200809L

#include <limits.h>
#include <math.h>
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
 * Pixels whose candidates are tested together: a chunk in which every
 * candidate is rejected is skipped at once.
 */
#define NLM_CHUNK 16

/*
 * Offsets processed together (1, 2, 4 or 8). Each group builds its integral
 * images in one pass and updates every pixel's accumulators once for all
 * its offsets.
 */
#ifndef NLM_BLOCK
#	define NLM_BLOCK 4
#endif


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
 * Used to fill the weight table, and directly when the table would be too
 * large.
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
 * @brief Denoises the rows [y0, y1) of the image.
 *
 * The (2S+1)^2 - 1 offsets of the search window are processed in groups of
 * NLM_BLOCK consecutive offsets (in the usual raster order). For a group:
 *
 * 1. The integral images of the squared differences of all its offsets are
 *    built in one pass, interleaved (entry j * NLM_BLOCK + k belongs to
 *    offset k). The running row sums of the offsets are independent, so the
 *    processor overlaps them instead of waiting on a single chain.
 *
 * 2. For every row, the patch sums of all offsets of the group are read
 *    with 4 lookups each, in a branch-free loop the compiler vectorizes.
 *    Chunks of NLM_CHUNK pixels in which every candidate of every offset is
 *    rejected are skipped at once. Otherwise each pixel's accumulators are
 *    loaded once, updated for the offsets of the group in order, and stored
 *    once: NLM_BLOCK times less accumulator traffic than one offset at a
 *    time, with the additions in the same order, hence the same result.
 *
 * The number of offsets, 4 S (S + 1), is a multiple of 8 for every S, so
 * groups of 1, 2, 4 or 8 offsets always divide it.
 *
 * The integral images hold 32-bit unsigned values that are allowed to wrap:
 * a patch sum is the difference of 4 entries, which is exact modulo 2^32,
 * and a patch sum itself is below 2^32 (at most 441 * 255^2).
 *
 * @param[in]  c    Context.
 * @param[in]  y0   First row of the band.
 * @param[in]  y1   One past the last row of the band.
 * @param[in]  ii   Scratch: NLM_BLOCK x (y1-y0 + 2P + 1) x (W + 2P + 1) values.
 * @param[in]  acc  Scratch: 3 x (y1-y0) x W floats (weight sum, weighted
 *                  value sum, maximum weight).
 * @param[in]  sums Scratch: NLM_BLOCK x W values (patch sums of one row).
 * @param[out] out  Output image (rows [y0, y1) are written).
 */
static void
integral_band(const NlmContext *c, long y0, long y1, uint32_t *ii, float *acc,
              uint32_t *sums, unsigned char *out)
{
	enum { K = NLM_BLOCK };
	const long W = c->w;
	const int P = c->p;
	const int S = c->s;
	const long R = P + S;
	const long PW = (long)c->pw;
	const long rows = y1 - y0;
	const long nv = rows + 2 * P;           /* rows of differences */
	const long nu = W + 2 * P;              /* columns of differences */
	const size_t iw = (size_t)nu + 1;       /* integral width (zero column) */
	const size_t ir = iw * K;               /* one interleaved integral row */
	const size_t n = (size_t)rows * (size_t)W;
	const uint32_t cutoff = (uint32_t)c->cutoff;
	const long side = 2 * S + 1;
	const long noff = side * side - 1;      /* offsets, center excluded */
	float *wsum = acc;
	float *vsum = acc + n;
	float *wmax = acc + 2 * n;

	memset(acc, 0, 3 * n * sizeof(float));
	memset(ii, 0, ir * sizeof(uint32_t)); /* zero row */

	for (long g = 0; g < noff; g += K) {
		int dx[K], dy[K];

		/* offsets g .. g+K-1 in raster order, skipping the center */
		for (int k = 0; k < K; k++) {
			long o = g + k;
			if (o >= noff / 2)
				o++;
			dy[k] = (int)(o / side) - S;
			dx[k] = (int)(o % side) - S;
		}

		/* 1. interleaved integral images of the group's squared differences */
		for (long i = 0; i < nv; i++) {
			const unsigned char *a = c->pad + (size_t)(y0 + R - P + i) * c->pw + (size_t)(R - P);
			const unsigned char *b[K];
			const uint32_t *prev = ii + (size_t)i * ir;
			uint32_t *cur = ii + (size_t)(i + 1) * ir;
			uint32_t rs[K];

			for (int k = 0; k < K; k++) {
				b[k] = a + (long)dy[k] * PW + dx[k];
				rs[k] = 0;
				cur[k] = 0;
			}

			for (long j = 0; j < nu; j++) {
				for (int k = 0; k < K; k++) {
					int d = (int)a[j] - (int)b[k][j];
					rs[k] += (uint32_t)(d * d);
					cur[(size_t)(j + 1) * K + k] = prev[(size_t)(j + 1) * K + k] + rs[k];
				}
			}
		}

		/* 2. patch sums, then weights, row by row */
		for (long y = 0; y < rows; y++) {
			const uint32_t *top = ii + (size_t)y * ir;
			const uint32_t *bot = ii + (size_t)(y + 2 * P + 1) * ir;
			const size_t shift = (size_t)(2 * P + 1) * K;
			const unsigned char *q[K];
			float *ws = wsum + (size_t)y * W;
			float *vs = vsum + (size_t)y * W;
			float *wm = wmax + (size_t)y * W;

			for (int k = 0; k < K; k++)
				q[k] = c->pad + (size_t)(y0 + y + R + dy[k]) * c->pw + (size_t)(R + dx[k]);

			/*
			 * All patch sums of the row for the whole group, branch-free
			 * and contiguous. The sums are below 2^31, so an unsigned
			 * comparison with the (non-negative) cutoff is the same as a
			 * signed one.
			 */
			for (size_t m = 0; m < (size_t)W * K; m++)
				sums[m] = bot[m + shift] - top[m + shift] - bot[m] + top[m];

			for (long x0 = 0; x0 < W; x0 += NLM_CHUNK) {
				long x1 = (x0 + NLM_CHUNK < W) ? x0 + NLM_CHUNK : W;
				uint32_t lo = UINT32_MAX;

				for (size_t m = (size_t)x0 * K; m < (size_t)x1 * K; m++)
					lo = (sums[m] < lo) ? sums[m] : lo;

				if (lo > cutoff)
					continue;

				for (long x = x0; x < x1; x++) {
					const uint32_t *sx = sums + (size_t)x * K;
					float a = ws[x], v = vs[x], mx = wm[x];

					/* same order as one offset at a time: k ascending */
					for (int k = 0; k < K; k++) {
						float w;

						if (sx[k] > cutoff)
							continue;

						w = lookup(c, (int)sx[k]);
						a += w;
						v += w * (float)q[k][x];
						if (w > mx)
							mx = w;
					}

					ws[x] = a;
					vs[x] = v;
					wm[x] = mx;
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
 * @brief Allocates scratch buffers for running bands.
 *
 * @param[in] width Largest image width the scratch will serve.
 * @param[in] band  Largest band height, in rows.
 * @param[in] patch Patch radius.
 *
 * @return Newly allocated scratch, or NULL on allocation failure.
 */
NlmScratch*
nlm_scratch_new(unsigned int width, long band, unsigned int patch)
{
	NlmScratch *s = calloc(1, sizeof(NlmScratch));
	const size_t iw = (size_t)width + 2 * patch + 1;

	if (!s)
		return NULL;

	s->width = width;
	s->band = band;
	s->patch = patch;
	s->ii = malloc((size_t)NLM_BLOCK * (size_t)(band + 2 * (long)patch + 1) * iw * sizeof(uint32_t));
	s->acc = malloc(3 * (size_t)band * (size_t)width * sizeof(float));
	s->sums = malloc((size_t)NLM_BLOCK * (size_t)width * sizeof(uint32_t));

	if (!s->ii || !s->acc || !s->sums) {
		nlm_scratch_free(s);
		return NULL;
	}

	return s;
}

/**
 * @brief Frees scratch buffers. Safe to call with NULL.
 *
 * @param[in,out] s Scratch.
 */
void
nlm_scratch_free(NlmScratch *s)
{
	if (!s)
		return;

	free(s->ii);
	free(s->acc);
	free(s->sums);
	free(s);
}

/**
 * @brief Prepares an image for denoising, band by band.
 *
 * Estimates the noise (unless given), pads the image, builds the weight
 * table and allocates the output. With @p parallel set, these steps use
 * the OpenMP threads.
 *
 * @param[in]  src       Noisy image (must outlive the job).
 * @param[in]  p         Filter parameters.
 * @param[in]  band      Rows per band (> 0).
 * @param[in]  parallel  Non-zero to parallelize the preparation.
 * @param[out] job       Job to initialize.
 * @param[out] sigma_out Noise standard deviation used (may be NULL).
 *
 * @return IMG_OK, or IMG_ERR_NOMEM / IMG_ERR_SIZE.
 */
int
nlm_job_prepare(const Image *src, const NlmParams *p, long band, int parallel,
                NlmJob *job, double *sigma_out)
{
	const long W = (long)src->width;
	const long H = (long)src->height;
	const double area = (double)(2 * p->patch + 1) * (double)(2 * p->patch + 1);
	NlmContext *c = &job->ctx;
	double sigma, h, cut;

	memset(job, 0, sizeof(*job));

	if (!src->data || W == 0 || H == 0 || band <= 0)
		return IMG_ERR_SIZE;

	sigma = (p->sigma >= 0.0) ? p->sigma : nlm_noise_estimate(src, parallel);
	if (sigma_out)
		*sigma_out = sigma;

	job->out.data = malloc((size_t)W * H);
	if (!job->out.data)
		return IMG_ERR_NOMEM;
	job->out.width = src->width;
	job->out.height = src->height;
	job->band = band;

	/* no noise, nothing to remove: no bands to run */
	h = p->h_factor * sigma;
	if (h <= 0.0) {
		memcpy(job->out.data, src->data, (size_t)W * H);
		job->bands = 0;
		return IMG_OK;
	}

	job->pad = pad_image(src, p->patch + p->search, parallel, &c->pw);
	if (!job->pad) {
		nlm_job_discard(job);
		return IMG_ERR_NOMEM;
	}

	c->pad = job->pad;
	c->w = W;
	c->h = H;
	c->p = (int)p->patch;
	c->s = (int)p->search;

	/*
	 * Work on patch sums instead of means: the exponent
	 *   (d^2 - 2 sigma^2) / h^2  with  d^2 = ssd / area
	 * equals (ssd - offset) * inv.
	 */
	c->offset = 2.0 * sigma * sigma * area;
	c->inv = 1.0 / (h * h * area);

	/*
	 * Largest patch sum whose weight is not negligible: arg <= NLM_MAX_ARG
	 * <=> ssd <= offset + NLM_MAX_ARG / inv. Kept as an integer so the
	 * inner loops compare integers; clamped to fit.
	 */
	cut = floor(c->offset + NLM_MAX_ARG / c->inv);
	c->cutoff = (cut >= (double)INT_MAX) ? INT_MAX : (int)cut;

	/* one expf() per possible patch sum instead of one per candidate */
	job->tab = build_table(c, parallel);
	c->wtab = job->tab;

	job->bands = (H + band - 1) / band;
	return IMG_OK;
}

/**
 * @brief Number of bands of a prepared job (0 if there is nothing to do).
 *
 * @param[in] job Prepared job.
 *
 * @return Number of bands.
 */
long
nlm_job_bands(const NlmJob *job)
{
	return job->bands;
}

/**
 * @brief Denoises one band of a prepared job.
 *
 * Bands of the same job, or of different jobs, may run concurrently on
 * different threads, each with its own scratch.
 *
 * @param[in,out] job     Prepared job.
 * @param[in]     b       Band index, 0 <= b < nlm_job_bands(job).
 * @param[in,out] scratch Scratch at least as wide as the image and as tall
 *                        as the band.
 */
void
nlm_job_run(NlmJob *job, long b, NlmScratch *scratch)
{
	const NlmContext *c = &job->ctx;
	long y0 = b * job->band;
	long y1 = (y0 + job->band < c->h) ? y0 + job->band : c->h;

	integral_band(c, y0, y1, scratch->ii, scratch->acc, scratch->sums,
	              job->out.data);
}

/**
 * @brief Completes a job: releases its buffers and hands over the output.
 *
 * @param[in,out] job Job whose bands have all run.
 * @param[out]    dst Denoised image (caller frees).
 */
void
nlm_job_finish(NlmJob *job, Image *dst)
{
	*dst = job->out;
	job->out.data = NULL;
	nlm_job_discard(job);
}

/**
 * @brief Releases a job without producing output. Safe on a zeroed job.
 *
 * @param[in,out] job Job.
 */
void
nlm_job_discard(NlmJob *job)
{
	free(job->pad);
	free(job->tab);
	image_free(&job->out);
	job->pad = NULL;
	job->tab = NULL;
	job->bands = 0;
}

/**
 * @brief Number of progress units of one image: its search offsets.
 *
 * The cost of denoising is proportional to the number of offsets of the
 * search window, 4 S (S + 1), so progress is counted in those units, split
 * over the bands of the image.
 *
 * @param[in] p Filter parameters.
 *
 * @return 4 S (S + 1).
 */
size_t
nlm_units(const NlmParams *p)
{
	return 4 * (size_t)p->search * ((size_t)p->search + 1);
}

/**
 * @brief Progress units of band @p b out of @p bands, for an image of
 *        @p units units: the shares add up to exactly @p units.
 */
size_t
nlm_band_units(size_t units, long b, long bands)
{
	if (bands <= 0)
		return 0;
	return units * (size_t)(b + 1) / (size_t)bands - units * (size_t)b / (size_t)bands;
}
