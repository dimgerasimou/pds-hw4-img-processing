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
 * Two methods are provided, with bit-for-bit identical output:
 *   direct    patches compared pixel by pixel, (2S+1)^2 (2P+1)^2 per pixel
 *   integral  patch sums from integral images of the squared differences,
 *             about (2S+1)^2 per pixel, independent of the patch size
 */

#ifndef NLM_H
#define NLM_H

#include "image.h"

/**
 * @enum NLM methods
 * @brief Ways of computing the patch distances (identical results).
 */
enum {
	NLM_INTEGRAL = 0, /**< Integral images, cost independent of the patch size */
	NLM_DIRECT        /**< Patches compared pixel by pixel */
};

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
	int method;          /**< NLM_INTEGRAL or NLM_DIRECT */
} NlmParams;

/* ------------------------------------------------------------------------- */
/*                            Public API Functions                           */
/* ------------------------------------------------------------------------- */

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
int nlm_denoise(const Image *src, Image *dst, const NlmParams *p, int parallel,
                double *sigma_out);

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

/**
 * @brief Returns the name of an NLM method ("direct", "integral").
 *
 * @param[in] method NLM_* value.
 *
 * @return Static name string.
 */
const char* nlm_method_name(int method);

/**
 * @brief Parses an NLM method name.
 *
 * @param[in] name Method name.
 *
 * @return NLM_* value, or -1 if unknown.
 */
int nlm_method_from_name(const char *name);

#endif /* NLM_H */
