/**
 * @file canny.h
 * @brief Canny edge detection.
 *
 * J. Canny, "A Computational Approach to Edge Detection", IEEE PAMI 1986:
 * Gaussian blur, Sobel gradients, non-maximum suppression, and hysteresis
 * (weak candidates are edges if 8-connected to a strong one). The result
 * replaces the image: 255 on edges, 0 elsewhere.
 *
 * On the CPU, steps 1-3 run on bands of rows (canny_band()) and hysteresis
 * once all bands of an image are done.
 */

#ifndef CANNY_H
#define CANNY_H

#include "canny_core.h"
#include "image.h"

typedef struct {
	double sigma; /* 0: no blur */
	double low;   /* thresholds on the gradient magnitude of the 8-bit image */
	double high;
} CannyParams;

typedef struct {
	int radius;
	int weights[2 * CANNY_MAX_RADIUS + 1]; /* sum to 256 */
	long long low2;                        /* squared, in internal units */
	long long high2;
} CannySetup;

/** @brief Computes the integer Gaussian weights and squared thresholds. */
void canny_setup(const CannyParams *p, CannySetup *s);

typedef struct {
	int *hb;
	int *b;
	int width;
	int band;
	int radius;
} CannyScratch;

CannyScratch* canny_scratch_new(int width, int band, int radius);
void canny_scratch_free(CannyScratch *sc);

/**
 * @brief Classifies rows [y0, y1) into CANNY_NONE / WEAK / EDGE in @p map.
 *
 * Bands may run concurrently, each thread with its own scratch.
 */
void canny_band(const Image *img, const CannySetup *s, int y0, int y1,
                unsigned char *map, CannyScratch *sc);

/** @brief Hysteresis on a class map, which then becomes the edge map (255/0). */
int canny_hysteresis(unsigned char *map, int w, int h);

#endif /* CANNY_H */
