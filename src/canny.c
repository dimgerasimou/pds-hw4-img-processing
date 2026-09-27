/**
 * @file canny.c
 * @brief CPU implementation of Canny edge detection.
 *
 * The per-pixel arithmetic is in canny_core.h, shared with the GPU. This
 * file computes the Gaussian weights and thresholds, classifies bands of
 * rows with small per-band buffers, and performs hysteresis as a flood fill
 * from the strong edges.
 */

#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "canny.h"

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
void
canny_setup(const CannyParams *p, CannySetup *s)
{
	double g[2 * CANNY_MAX_RADIUS + 1], total = 0.0;
	long long lo, hi;
	int r, sum = 0;

	memset(s, 0, sizeof(*s));

	r = (p->sigma > 0.0) ? (int)ceil(3.0 * p->sigma) : 0;
	if (r > CANNY_MAX_RADIUS)
		r = CANNY_MAX_RADIUS;
	s->radius = r;

	for (int i = -r; i <= r; i++) {
		g[i + r] = (r > 0) ? exp(-(double)i * i / (2.0 * p->sigma * p->sigma)) : 1.0;
		total += g[i + r];
	}

	for (int i = 0; i <= 2 * r; i++) {
		s->weights[i] = (int)lround(256.0 * g[i] / total);
		sum += s->weights[i];
	}

	/* rounding may miss 256 by a little: correct the center weight */
	s->weights[r] += 256 - sum;

	/* internal gradients are in units of 1/256 of the 8-bit image */
	lo = llround(p->low * 256.0);
	hi = llround(p->high * 256.0);
	s->low2 = lo * lo;
	s->high2 = hi * hi;
}

/**
 * @brief Allocates scratch for bands of up to @p band rows of images up to
 *        @p width pixels wide.
 *
 * @return Newly allocated scratch, or NULL on allocation failure.
 */
CannyScratch*
canny_scratch_new(int width, int band, int radius)
{
	CannyScratch *sc = calloc(1, sizeof(CannyScratch));

	if (!sc)
		return NULL;

	sc->width = width;
	sc->band = band;
	sc->radius = radius;

	/* b: band + 2 rows on each side (Sobel, and Sobel at the NMS neighbors) */
	sc->b = malloc((size_t)(band + 4) * width * sizeof(int));
	/* hb: rows of b + the Gaussian radius on each side */
	sc->hb = malloc((size_t)(band + 4 + 2 * radius) * width * sizeof(int));

	if (!sc->b || !sc->hb) {
		canny_scratch_free(sc);
		return NULL;
	}

	return sc;
}

/**
 * @brief Frees scratch. Safe to call with NULL.
 */
void
canny_scratch_free(CannyScratch *sc)
{
	if (!sc)
		return;

	free(sc->hb);
	free(sc->b);
	free(sc);
}

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
 * Every row an access mirrors to stays within the buffers: mirroring moves
 * an index only across the image border, which the windows reach whenever
 * such an index occurs.
 *
 * @param[in]  img Image (read only).
 * @param[in]  s   Setup from canny_setup().
 * @param[in]  y0  First row of the band.
 * @param[in]  y1  One past the last row.
 * @param[out] map Class map of the whole image (rows [y0, y1) are written).
 * @param[in]  sc  Scratch at least as wide as the image and as tall as the band.
 */
void
canny_band(const Image *img, const CannySetup *s, int y0, int y1,
           unsigned char *map, CannyScratch *sc)
{
	const int w = (int)img->width, h = (int)img->height, r = s->radius;
	/* blurred rows needed: the band +- 2; horizontal sums: those +- r */
	const int lo = (y0 - 2 > 0) ? y0 - 2 : 0;
	const int hi = (y1 + 2 < h) ? y1 + 2 : h;
	const int hlo = (lo - r > 0) ? lo - r : 0;
	const int hhi = (hi + r < h) ? hi + r : h;

	for (int y = hlo; y < hhi; y++)
		for (int x = 0; x < w; x++)
			sc->hb[(size_t)(y - hlo) * w + x] = canny_blur_h(img->data, w, x, y, s->weights, r);

	for (int y = lo; y < hi; y++)
		for (int x = 0; x < w; x++)
			sc->b[(size_t)(y - lo) * w + x] = canny_blur_v(sc->hb, hlo, w, h, x, y, s->weights, r);

	for (int y = y0; y < y1; y++)
		for (int x = 0; x < w; x++)
			map[(size_t)y * w + x] = canny_classify(sc->b, lo, w, h, x, y, s->low2, s->high2);
}

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
int
canny_hysteresis(unsigned char *map, int w, int h)
{
	const size_t n = (size_t)w * h;
	size_t *stack = malloc(n * sizeof(size_t));
	size_t top = 0;

	if (!stack)
		return IMG_ERR_NOMEM;

	for (size_t i = 0; i < n; i++)
		if (map[i] == CANNY_EDGE)
			stack[top++] = i;

	/* every pixel is pushed at most once: when it turns from weak to edge */
	while (top > 0) {
		const size_t i = stack[--top];
		const int x = (int)(i % (size_t)w), y = (int)(i / (size_t)w);

		for (int dy = -1; dy <= 1; dy++) {
			for (int dx = -1; dx <= 1; dx++) {
				const int nx = x + dx, ny = y + dy;
				size_t j;

				if (nx < 0 || ny < 0 || nx >= w || ny >= h)
					continue;

				j = (size_t)ny * w + nx;
				if (map[j] == CANNY_WEAK) {
					map[j] = CANNY_EDGE;
					stack[top++] = j;
				}
			}
		}
	}

	free(stack);

	for (size_t i = 0; i < n; i++)
		map[i] = (map[i] == CANNY_EDGE) ? 255 : 0;

	return IMG_OK;
}
