/**
 * @file nlm.h
 * @brief Non-Local Means denoising.
 *
 * Implements the pixelwise Non-Local Means filter as described in
 * A. Buades, B. Coll, J.-M. Morel, "Non-Local Means Denoising",
 * Image Processing On Line, 1 (2011), pp. 208-212.
 *
 * Every output pixel p is a weighted average of the pixels q in a search
 * window around it. The weight measures how similar the patches around p
 * and q are:
 *
 *   d^2(p,q) = mean of the squared differences of the two patches
 *   w(p,q)   = exp( -max(d^2 - 2 sigma^2, 0) / h^2 ),   h = k * sigma
 *
 * Two noisy copies of the same patch differ by 2 sigma^2 on average, so
 * that amount is subtracted before weighting. The center pixel gets the
 * largest weight found among its neighbors rather than 1, which would
 * otherwise dominate the average.
 *
 * The noise standard deviation sigma is either given or estimated per image
 * (J. Immerkaer, "Fast Noise Variance Estimation", CVIU 64(2), 1996).
 *
 * Borders are handled by mirroring the image. Every output pixel depends
 * only on the input, so the result is identical for any thread count.
 *
 * Patch distances are computed with integral images, at a cost of about
 * (2S+1)^2 per pixel independent of the patch size, and the work of an
 * image is split into bands of rows that can run on any thread: see
 * nlm_job_prepare().
 */

#ifndef NLM_H
#define NLM_H

#include <stdint.h>

#include "image.h"
#include "progress.h"

/* ------------------------------------------------------------------------- */
/*                              Data Structures                              */
/* ------------------------------------------------------------------------- */

/**
 * @struct NlmParams
 * @brief Parameters of the NLM filter.
 */
typedef struct {
	unsigned int patch;  /**< Patch radius: patches are (2r+1) x (2r+1) */
	unsigned int search; /**< Search radius: window is (2r+1) x (2r+1) */
	double h_factor;     /**< Filtering strength k, with h = k * sigma */
	double sigma;        /**< Noise standard deviation, < 0: estimate per image */
} NlmParams;

/**
 * @struct NlmContext
 * @brief Per-image values shared by all bands (internal).
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

/**
 * @struct NlmJob
 * @brief One image being denoised band by band.
 *
 * A job splits the work of an image into bands of rows that can run on any
 * thread, in any order: nlm_job_prepare(), then nlm_job_run() for every
 * band, then nlm_job_finish(). Treat as opaque.
 */
typedef struct {
	NlmContext ctx;      /**< Shared per-image values */
	unsigned char *pad;  /**< Owned padded input */
	float *tab;          /**< Owned weight table, or NULL */
	Image out;           /**< Output being filled */
	long band;           /**< Rows per band */
	long bands;          /**< Number of bands (0: nothing to run) */
} NlmJob;

/**
 * @struct NlmScratch
 * @brief Per-thread working memory for running bands.
 */
typedef struct {
	unsigned int width;  /**< Largest image width served */
	long band;           /**< Largest band height served */
	unsigned int patch;  /**< Patch radius */
	uint32_t *ii;        /**< Integral image of one band */
	float *acc;          /**< Weight accumulators of one band */
	uint32_t *sums;      /**< Patch sums of one row */
} NlmScratch;

/* ------------------------------------------------------------------------- */
/*                            Public API Functions                           */
/* ------------------------------------------------------------------------- */

/**
 * @brief Allocates scratch buffers for running bands.
 *
 * @param[in] width Largest image width the scratch will serve.
 * @param[in] band  Largest band height, in rows.
 * @param[in] patch Patch radius.
 *
 * @return Newly allocated scratch, or NULL on allocation failure.
 */
NlmScratch* nlm_scratch_new(unsigned int width, long band, unsigned int patch);

/**
 * @brief Frees scratch buffers. Safe to call with NULL.
 *
 * @param[in,out] s Scratch.
 */
void nlm_scratch_free(NlmScratch *s);

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
int nlm_job_prepare(const Image *src, const NlmParams *p, long band, int parallel,
                    NlmJob *job, double *sigma_out);

/**
 * @brief Number of bands of a prepared job (0 if there is nothing to do).
 *
 * @param[in] job Prepared job.
 *
 * @return Number of bands.
 */
long nlm_job_bands(const NlmJob *job);

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
void nlm_job_run(NlmJob *job, long b, NlmScratch *scratch);

/**
 * @brief Completes a job: releases its buffers and hands over the output.
 *
 * @param[in,out] job Job whose bands have all run.
 * @param[out]    dst Denoised image (caller frees).
 */
void nlm_job_finish(NlmJob *job, Image *dst);

/**
 * @brief Releases a job without producing output. Safe on a zeroed job.
 *
 * @param[in,out] job Job.
 */
void nlm_job_discard(NlmJob *job);

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
size_t nlm_units(const NlmParams *p);

/**
 * @brief Progress units of band @p b out of @p bands, for an image of
 *        @p units units: the shares add up to exactly @p units.
 */
size_t nlm_band_units(size_t units, long b, long bands);

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
double nlm_noise_estimate(const Image *img, int parallel);

#endif /* NLM_H */
