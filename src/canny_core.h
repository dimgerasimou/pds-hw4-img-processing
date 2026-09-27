/**
 * @file canny_core.h
 * @brief Per-pixel Canny arithmetic shared by the CPU and GPU implementations.
 *
 * These inline functions are compiled into both the C code (canny.c) and
 * the CUDA kernels (gpu.cu), so that both backends perform exactly the same
 * operations. Everything is integer arithmetic, so the results are exact
 * and the two backends produce bit-identical edge maps:
 *
 *   - Gaussian blur: separable, with integer weights summing to 256; the
 *     horizontal pass keeps the full sum (at most 255 * 256), the vertical
 *     pass rounds to units of 1/256 gray level (at most 65280);
 *   - Sobel gradients on the blurred image (at most 4 * 65280 each);
 *   - gradient magnitudes compared squared, as 64-bit integers;
 *   - gradient direction quantized to 4 sectors by exact comparisons of
 *     |gy| / |gx| with tan(22.5 deg) and tan(67.5 deg), using
 *     tan(22.5 deg) ~ CANNY_TAN / CANNY_ONE.
 *
 * Image borders are mirrored (-1 -> 0, n -> n-1), as for NLM.
 */

#ifndef CANNY_CORE_H
#define CANNY_CORE_H

#include <stddef.h>

#ifdef __CUDACC__
#	define CANNY_HD __host__ __device__
#else
#	define CANNY_HD
#endif

/* Largest Gaussian radius: ceil(3 sigma) for sigma up to 10 */
#define CANNY_MAX_RADIUS 30

/* tan(22.5 deg) = sqrt(2) - 1, in units of 1/65536 */
#define CANNY_TAN 27146LL
#define CANNY_ONE 65536LL

/* Pixel classes after non-maximum suppression and thresholding */
enum {
	CANNY_NONE = 0, /**< Not an edge */
	CANNY_WEAK = 1, /**< Edge only if connected to a strong edge */
	CANNY_EDGE = 2  /**< Edge (strong, or weak connected to strong) */
};

/**
 * @brief Mirrors an index into [0, n) (symmetric: -1 -> 0, n -> n-1).
 */
static inline CANNY_HD int
canny_reflect(int i, int n)
{
	const int period = 2 * n;

	i %= period;
	if (i < 0)
		i += period;
	if (i >= n)
		i = period - 1 - i;
	return i;
}

/*
 * The functions below take the array they read (horizontal sums hb or
 * blurred image b) together with the image row its first row holds
 * (@p first), so that the CPU can work on bands with small buffers; the GPU
 * passes whole images (first = 0). Away from the image border, where no
 * index needs mirroring, they take a fast path without canny_reflect(); the
 * values and their order are the same, so the results are too.
 */

/**
 * @brief Horizontal Gaussian pass at (x, y): sum of weighted pixels.
 */
static inline CANNY_HD int
canny_blur_h(const unsigned char *img, int w, int x, int y, const int *wt, int r)
{
	const unsigned char *row = img + (size_t)y * w;
	int sum = 0;

	if (x >= r && x + r < w) {
		const unsigned char *p = row + x - r;
		for (int i = 0; i <= 2 * r; i++)
			sum += (int)p[i] * wt[i];
	} else {
		for (int i = -r; i <= r; i++)
			sum += (int)row[canny_reflect(x + i, w)] * wt[i + r];
	}
	return sum;
}

/**
 * @brief Vertical Gaussian pass at (x, y) over the horizontal sums, rounded
 *        to units of 1/256 gray level.
 *
 * @param hb    Horizontal sums, rows first, first + 1, ... of the image.
 * @param first Image row held by the first row of @p hb.
 */
static inline CANNY_HD int
canny_blur_v(const int *hb, int first, int w, int h, int x, int y, const int *wt, int r)
{
	int sum = 0;

	if (y >= r && y + r < h) {
		const int *p = hb + (size_t)(y - r - first) * w + x;
		for (int i = 0; i <= 2 * r; i++)
			sum += p[(size_t)i * w] * wt[i];
	} else {
		for (int i = -r; i <= r; i++)
			sum += hb[(size_t)(canny_reflect(y + i, h) - first) * w + x] * wt[i + r];
	}
	return (sum + 128) >> 8;
}

/**
 * @brief Sobel gradient of the blurred image at (x, y).
 *
 * @param b     Blurred image, rows first, first + 1, ... of the image.
 * @param first Image row held by the first row of @p b.
 * @param[out] mag2 Squared gradient magnitude.
 * @param[out] dir  Direction sector (see canny_classify()), may be NULL.
 */
static inline CANNY_HD void
canny_sobel(const int *b, int first, int w, int h, int x, int y, long long *mag2, int *dir)
{
	const int inside = (x > 0 && x < w - 1 && y > 0 && y < h - 1);
	const int xm = inside ? x - 1 : canny_reflect(x - 1, w);
	const int xp = inside ? x + 1 : canny_reflect(x + 1, w);
	const int ym = inside ? y - 1 : canny_reflect(y - 1, h);
	const int yp = inside ? y + 1 : canny_reflect(y + 1, h);
	const int *rm = b + (size_t)(ym - first) * w;
	const int *r0 = b + (size_t)(y - first) * w;
	const int *rp = b + (size_t)(yp - first) * w;
	const int gx = (rm[xp] + 2 * r0[xp] + rp[xp]) - (rm[xm] + 2 * r0[xm] + rp[xm]);
	const int gy = (rp[xm] + 2 * rp[x] + rp[xp]) - (rm[xm] + 2 * rm[x] + rm[xp]);
	const long long ax = gx < 0 ? -(long long)gx : gx;
	const long long ay = gy < 0 ? -(long long)gy : gy;

	*mag2 = (long long)gx * gx + (long long)gy * gy;

	if (!dir)
		return;

	/*
	 * 0: gradient near horizontal  (|gy|/|gx| <= tan 22.5)
	 * 2: gradient near vertical    (|gy|/|gx| >= tan 67.5 = 1 / tan 22.5)
	 * 1: diagonal, gx and gy of the same sign (down-right in image coords)
	 * 3: diagonal, opposite signs
	 */
	if (ay * CANNY_ONE <= ax * CANNY_TAN)
		*dir = 0;
	else if (ay * CANNY_TAN >= ax * CANNY_ONE)
		*dir = 2;
	else
		*dir = ((gx > 0) == (gy > 0)) ? 1 : 3;
}

/**
 * @brief Non-maximum suppression and double thresholding at (x, y).
 *
 * A pixel survives if its gradient is at least the low threshold and a
 * local maximum along the gradient direction: strictly greater than the
 * neighbor on one side and at least the neighbor on the other, so that a
 * ridge two pixels wide keeps exactly one of them.
 *
 * @param b     Blurred image, rows first, first + 1, ... of the image.
 * @param first Image row held by the first row of @p b.
 * @param tl2   Squared low threshold (in the units of canny_sobel()).
 * @param th2   Squared high threshold.
 *
 * @return CANNY_NONE, CANNY_WEAK or CANNY_EDGE.
 */
static inline CANNY_HD unsigned char
canny_classify(const int *b, int first, int w, int h, int x, int y, long long tl2, long long th2)
{
	long long m, n1, n2;
	int dir, x1, y1, x2, y2;

	canny_sobel(b, first, w, h, x, y, &m, &dir);
	if (m == 0 || m < tl2)
		return CANNY_NONE;

	switch (dir) {
	case 0:  x1 = x - 1; y1 = y;     x2 = x + 1; y2 = y;     break;
	case 2:  x1 = x;     y1 = y - 1; x2 = x;     y2 = y + 1; break;
	case 1:  x1 = x - 1; y1 = y - 1; x2 = x + 1; y2 = y + 1; break;
	default: x1 = x + 1; y1 = y - 1; x2 = x - 1; y2 = y + 1; break;
	}

	canny_sobel(b, first, w, h, canny_reflect(x1, w), canny_reflect(y1, h), &n1, NULL);
	canny_sobel(b, first, w, h, canny_reflect(x2, w), canny_reflect(y2, h), &n2, NULL);

	if (!(m > n1 && m >= n2))
		return CANNY_NONE;

	return (m >= th2) ? CANNY_EDGE : CANNY_WEAK;
}

#endif /* CANNY_CORE_H */
