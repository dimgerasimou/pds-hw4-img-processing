/**
 * @file nlm.c
 * @brief Non-Local Means denoising.
 *
 * Patch distances use integral images (Darbon et al., ISBI 2008): for each
 * offset of the search window, the squared differences between the image
 * and its shifted copy go into a summed-area table, from which any patch sum
 * is 4 lookups. The image is padded by mirroring, so the loops need no
 * bounds checks.
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

/* exp(-30) ~ 1e-13: smaller weights cannot change an 8-bit result */
#define NLM_MAX_ARG 30.0

/* Beyond this (strong noise, big patches), weights are computed directly. */
#define NLM_MAX_TABLE (1L << 22)

/* Chunk of pixels skipped at once when all its candidates are rejected. */
#define NLM_CHUNK 16

/* Offsets per integral pass: 1, 2, 4 or 8 (4 S (S+1) is a multiple of 8). */
#ifndef NLM_BLOCK
#	define NLM_BLOCK 4
#endif

/* Symmetric mirroring (-1 -> 0, n -> n-1), for any distance outside [0, n). */
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

static inline float
weight(const NlmContext *c, int ssd)
{
	return nlm_weight(ssd, c->off_f, c->scale_f);
}

/* Sums beyond the cutoff map to the table's last entry, which is 0. */
static inline float
lookup(const NlmContext *c, int ssd)
{
	if (c->wtab)
		return c->wtab[ssd > c->cutoff ? c->cutoff + 1 : ssd];

	return (ssd > c->cutoff) ? 0.0f : weight(c, ssd);
}

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

/* The center counts as much as its most similar neighbor. */
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

/*
 * Groups of NLM_BLOCK offsets build their integral images in one
 * interleaved pass (entry j * NLM_BLOCK + k belongs to offset k), then every
 * pixel's accumulators are loaded once per group and updated for its offsets
 * in order, so the result matches processing one offset at a time.
 *
 * The integral images are allowed to wrap: a patch sum is the difference of
 * 4 entries, exact modulo 2^32, and at most 441 * 255^2 itself.
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
	const long nv = rows + 2 * P;      /* rows of differences */
	const long nu = W + 2 * P;         /* columns of differences */
	const size_t iw = (size_t)nu + 1;  /* with a zero column */
	const size_t ir = iw * K;
	const size_t n = (size_t)rows * (size_t)W;
	const uint32_t cutoff = (uint32_t)c->cutoff;
	const long side = 2 * S + 1;
	const long noff = side * side - 1;
	float *wsum = acc;
	float *vsum = acc + n;
	float *wmax = acc + 2 * n;

	memset(acc, 0, 3 * n * sizeof(float));
	memset(ii, 0, ir * sizeof(uint32_t));

	for (long g = 0; g < noff; g += K) {
		int dx[K], dy[K];

		/* raster order, skipping the center */
		for (int k = 0; k < K; k++) {
			long o = g + k;
			if (o >= noff / 2)
				o++;
			dy[k] = (int)(o / side) - S;
			dx[k] = (int)(o % side) - S;
		}

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
			 * Branch-free so that it vectorizes. Sums are below 2^31, so the
			 * unsigned comparison with the cutoff is safe.
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

					/* k ascending: same order as one offset at a time */
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

double
nlm_noise_estimate(const Image *img, int parallel)
{
	const long W = (long)img->width;
	const long H = (long)img->height;
	unsigned long long sum = 0;

	if (W < 3 || H < 3)
		return 0.0;

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

	sigma = (p->sigma >= 0.0) ? p->sigma : p->sigma_scale * nlm_noise_estimate(src, parallel);
	if (sigma_out)
		*sigma_out = sigma;

	job->out.data = malloc((size_t)W * H);
	if (!job->out.data)
		return IMG_ERR_NOMEM;
	job->out.width = src->width;
	job->out.height = src->height;
	job->band = band;

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

	/* in patch sums: (d^2 - 2 sigma^2) / h^2 == (ssd - offset) * inv */
	c->offset = 2.0 * sigma * sigma * area;
	c->inv = 1.0 / (h * h * area);
	c->off_f = (float)c->offset;
	c->scale_f = (float)(c->inv * 1.4426950408889634); /* 1 / ln 2 */

	/* weights are negligible beyond; an int, so the loops compare integers */
	cut = floor(c->offset + NLM_MAX_ARG / c->inv);
	c->cutoff = (cut >= (double)INT_MAX) ? INT_MAX : (int)cut;

	job->tab = build_table(c, parallel);
	c->wtab = job->tab;

	job->bands = (H + band - 1) / band;
	return IMG_OK;
}

long
nlm_job_bands(const NlmJob *job)
{
	return job->bands;
}

void
nlm_job_run(NlmJob *job, long b, NlmScratch *scratch)
{
	const NlmContext *c = &job->ctx;
	long y0 = b * job->band;
	long y1 = (y0 + job->band < c->h) ? y0 + job->band : c->h;

	integral_band(c, y0, y1, scratch->ii, scratch->acc, scratch->sums,
	              job->out.data);
}

void
nlm_job_finish(NlmJob *job, Image *dst)
{
	*dst = job->out;
	job->out.data = NULL;
	nlm_job_discard(job);
}

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

size_t
nlm_units(const NlmParams *p)
{
	return 4 * (size_t)p->search * ((size_t)p->search + 1);
}

size_t
nlm_band_units(size_t units, long b, long bands)
{
	if (bands <= 0)
		return 0;
	return units * (size_t)(b + 1) / (size_t)bands - units * (size_t)b / (size_t)bands;
}
