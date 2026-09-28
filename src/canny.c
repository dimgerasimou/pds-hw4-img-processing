/**
 * @file canny.c
 * @brief Canny edge detection on the CPU.
 */

#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "canny.h"

/*
 * Above any gradient of an 8-bit image (4 * 255 * sqrt(2) < 1500), so it
 * still means "no edges", but keeps the squared thresholds from overflowing.
 */
#define MAX_THRESHOLD 1e6

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
	lo = llround(fmin(p->low, MAX_THRESHOLD) * 256.0);
	hi = llround(fmin(p->high, MAX_THRESHOLD) * 256.0);
	s->low2 = lo * lo;
	s->high2 = hi * hi;
}

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

void
canny_scratch_free(CannyScratch *sc)
{
	if (!sc)
		return;

	free(sc->hb);
	free(sc->b);
	free(sc);
}

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

int
canny_hysteresis(unsigned char *map, int w, int h)
{
	const size_t n = (size_t)w * h;
	size_t *stack, top = 0, cand = 0;

	/* only candidates are ever pushed, each at most once */
	for (size_t i = 0; i < n; i++)
		cand += (map[i] != CANNY_NONE);

	stack = malloc((cand ? cand : 1) * sizeof(size_t));
	if (!stack)
		return IMG_ERR_NOMEM;

	for (size_t i = 0; i < n; i++)
		if (map[i] == CANNY_EDGE)
			stack[top++] = i;

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
