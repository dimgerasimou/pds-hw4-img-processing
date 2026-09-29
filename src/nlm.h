/**
 * @file nlm.h
 * @brief Non-Local Means denoising.
 *
 * A. Buades, B. Coll, J.-M. Morel, "Non-Local Means Denoising", IPOL 2011.
 *
 *   d^2(p,q) = mean squared difference of the patches around p and q
 *   w(p,q)   = exp(-max(d^2 - 2 sigma^2, 0) / h^2),   h = k * sigma
 *
 * The center pixel gets the largest weight among its neighbors instead of 1.
 * Sigma is given or estimated per image (Immerkaer, CVIU 1996). Patch
 * distances come from integral images, so the cost does not depend on the
 * patch size. Each image is split into bands of rows that any thread can
 * run: nlm_job_prepare(), nlm_job_run() per band, nlm_job_finish().
 */

#ifndef NLM_H
#define NLM_H

#include <stdint.h>

#include "image.h"
#include "nlm_core.h"
#include "progress.h"

typedef struct {
	unsigned int patch;  /* radius */
	unsigned int search; /* radius */
	double h_factor;     /* h = h_factor * sigma; 0: automatic, see nlm_noise_estimate() */
	double sigma;        /* < 0: estimate per image */
	double sigma_scale;  /* multiplies the estimate */
} NlmParams;

/* Per-image values shared by all bands of a job. */
typedef struct {
	const unsigned char *pad; /* mirrored input, padded by p + s */
	size_t pw;
	long w;
	long h;
	int p;
	int s;
	double offset;            /* 2 sigma^2 * patch area */
	double inv;               /* 1 / (h^2 * patch area) */
	float off_f, scale_f;     /* arguments of nlm_weight(): offset, inv / ln 2 */
	int cutoff;               /* largest patch sum with a non-negligible weight */
	const float *wtab;        /* CPU: weights for patch sums 0..cutoff+1, or NULL */
} NlmContext;

typedef struct {
	NlmContext ctx;
	unsigned char *pad;
	float *tab;
	Image out;
	long band;           /* rows per band */
	long bands;          /* 0: nothing to do, out is a copy of the input */
} NlmJob;

/* Per-thread working memory for nlm_job_run(). */
typedef struct {
	unsigned int width;
	long band;
	unsigned int patch;
	uint32_t *ii;
	float *acc;
	uint32_t *sums;
} NlmScratch;

NlmScratch* nlm_scratch_new(unsigned int width, long band, unsigned int patch);
void nlm_scratch_free(NlmScratch *s);

/**
 * @brief Prepares an image for denoising.
 *
 * Estimates the noise unless given, pads the image, builds the weight table
 * and allocates the output. With @p parallel set, these steps use all
 * threads. @p src must outlive the job.
 *
 * Without p->h_factor, the strength is 0.7 for white noise, and rises to 1.2
 * and then 1.6 as the excess of the noise estimate grows (see
 * nlm_noise_estimate()): correlated noise makes patch distances fluctuate
 * more and needs a larger tolerance. @p sigma_out and @p strength_out
 * (h / sigma) receive what was used; either may be NULL.
 */
int nlm_job_prepare(const Image *src, const NlmParams *p, long band, int parallel,
                    NlmJob *job, double *sigma_out, double *strength_out);

long nlm_job_bands(const NlmJob *job);

/**
 * @brief Denoises band @p b of a job.
 *
 * Bands may run concurrently, each thread with its own scratch.
 */
void nlm_job_run(NlmJob *job, long b, NlmScratch *scratch);

/** @brief Hands the output over to @p dst and releases the job. */
void nlm_job_finish(NlmJob *job, Image *dst);

/** @brief Releases a job without output. Safe on a zeroed job. */
void nlm_job_discard(NlmJob *job);

/**
 * @brief Progress units of one image: the offsets of the search window,
 *        4 S (S + 1), to which the cost is proportional.
 */
size_t nlm_units(const NlmParams *p);

/** @brief Share of @p units for band @p b; the shares add up to @p units. */
size_t nlm_band_units(size_t units, long b, long bands);

/** @brief What nlm_noise_estimate() finds out about the noise of an image. */
typedef struct {
	double sigma;    /**< standard deviation, in gray levels */
	double excess;   /**< the flat-block estimate over Immerkaer's: about 1 or less for white noise */
} NlmNoise;

/**
 * @brief Estimates the noise of an image.
 *
 * The larger of two estimates. Immerkaer's measures the finest scale: exact
 * for white noise, but it reads correlated noise (CT reconstructions, for
 * one) several times too low. The second takes the variance of the flattest
 * 16x16 blocks after a plane fit, which includes noise of any correlation;
 * it is scaled by 0.9 to leave white noise to the first, and ignored below 5
 * gray levels, where noise cannot be told from the texture of the image. excess is the
 * ratio of the two, 1 if no flat blocks could be found: well above 1, the
 * noise is not white, and the filter needs a larger tolerance h (see
 * nlm_job_prepare()). Texture counts partly as noise. Does not depend on the
 * thread count. Returns zeros for images smaller than 3x3.
 */
NlmNoise nlm_noise_estimate(const Image *img, int parallel);

#endif /* NLM_H */
