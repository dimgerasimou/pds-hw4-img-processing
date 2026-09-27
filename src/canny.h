/**
 * @file canny.h
 * @brief Canny edge detection.
 *
 * J. Canny, "A Computational Approach to Edge Detection", IEEE PAMI 8(6),
 * 1986. Four steps:
 *
 *   1. Gaussian blur of standard deviation sigma, to suppress noise;
 *   2. Sobel gradients of the blurred image;
 *   3. non-maximum suppression: a pixel stays a candidate only if its
 *      gradient is a local maximum along the gradient direction, which
 *      thins edges to one pixel;
 *   4. hysteresis: candidates above the high threshold are edges, and
 *      candidates above the low threshold are edges if they are connected
 *      (8-neighborhood) to an edge.
 *
 * The output replaces the image: 255 on edges, 0 elsewhere. All arithmetic
 * is integer (see canny_core.h), shared with the CUDA implementation, so the
 * CPU and GPU produce identical edge maps.
 *
 * On the CPU, steps 1-3 run on bands of rows (canny_band()), which any
 * thread can process independently, and step 4 (canny_hysteresis()) once
 * all bands of an image are done.
 */

#ifndef CANNY_H
#define CANNY_H

#include "canny_core.h"
#include "image.h"

/* ------------------------------------------------------------------------- */
/*                              Data Structures                              */
/* ------------------------------------------------------------------------- */

/**
 * @struct CannyParams
 * @brief User parameters of the edge detector.
 */
typedef struct {
	double sigma; /**< Gaussian standard deviation (0: no blur) */
	double low;   /**< Low threshold, gradient magnitude of the 8-bit image */
	double high;  /**< High threshold, gradient magnitude of the 8-bit image */
} CannyParams;

/**
 * @struct CannySetup
 * @brief Values derived from the parameters, computed once.
 */
typedef struct {
	int radius;                          /**< Gaussian radius, ceil(3 sigma) */
	int weights[2 * CANNY_MAX_RADIUS + 1]; /**< Integer weights, sum 256 */
	long long low2;                      /**< Squared low threshold, internal units */
	long long high2;                     /**< Squared high threshold, internal units */
} CannySetup;

/* ------------------------------------------------------------------------- */
/*                            Public API Functions                           */
/* ------------------------------------------------------------------------- */

/**
 * @brief Derives the Gaussian weights and squared thresholds.
 *
 * The weights exp(-i^2 / (2 sigma^2)) are scaled to integers summing to
 * exactly 256. Thresholds are converted to the internal gradient units (the
 * blurred image is in units of 1/256 gray level) and squared.
 *
 * @param[in]  p Parameters (0 <= sigma <= 10, 0 <= low <= high).
 * @param[out] s Setup.
 */
void canny_setup(const CannyParams *p, CannySetup *s);

/**
 * @struct CannyScratch
 * @brief Per-thread working memory for canny_band().
 */
typedef struct {
	int *hb;        /**< Horizontal sums of the band plus its border rows */
	int *b;         /**< Blurred image of the band plus its border rows */
	int width;      /**< Largest image width served */
	int band;       /**< Largest band height served */
	int radius;     /**< Gaussian radius */
} CannyScratch;

/**
 * @brief Allocates scratch for bands of up to @p band rows of images up to
 *        @p width pixels wide.
 *
 * @return Newly allocated scratch, or NULL on allocation failure.
 */
CannyScratch* canny_scratch_new(int width, int band, int radius);

/**
 * @brief Frees scratch. Safe to call with NULL.
 */
void canny_scratch_free(CannyScratch *sc);

/**
 * @brief Classifies the rows [y0, y1) of an image (steps 1-3).
 *
 * Computes the blur and gradients the band needs, including a border of
 * rows above and below it, in the scratch buffers, and writes CANNY_NONE /
 * CANNY_WEAK / CANNY_EDGE for every pixel of the band into @p map. Bands of
 * the same image, or of different images, may run concurrently with
 * separate scratch. The band's border rows are computed again by the
 * neighboring bands; every value is the same whichever band computes it.
 *
 * @param[in]  img Image (read only).
 * @param[in]  s   Setup from canny_setup().
 * @param[in]  y0  First row of the band.
 * @param[in]  y1  One past the last row.
 * @param[out] map Class map of the whole image (rows [y0, y1) are written).
 * @param[in]  sc  Scratch at least as wide as the image and as tall as the band.
 */
void canny_band(const Image *img, const CannySetup *s, int y0, int y1,
                unsigned char *map, CannyScratch *sc);

/**
 * @brief Hysteresis on a map of CANNY_NONE / CANNY_WEAK / CANNY_EDGE values:
 *        every weak pixel 8-connected to an edge through weak pixels becomes
 *        an edge; then the map becomes the edge map (255 / 0).
 *
 * @param[in,out] map Class map, w x h; on return the edge map.
 * @param[in]     w   Width.
 * @param[in]     h   Height.
 *
 * @return IMG_OK or IMG_ERR_NOMEM.
 */
int canny_hysteresis(unsigned char *map, int w, int h);

#endif /* CANNY_H */
