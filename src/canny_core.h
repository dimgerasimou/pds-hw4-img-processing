/**
 * @file canny_core.h
 * @brief Per-pixel Canny arithmetic, compiled into both the C code and the
 *        CUDA kernels.
 *
 * All integer, so both backends produce identical edge maps: blur weights
 * sum to 256, the blurred image is in units of 1/256 gray level, gradient
 * magnitudes are compared squared, and directions are quantized by exact
 * comparisons with tan 22.5 deg.
 *
 * The functions take the array they read together with the image row held
 * by its first row (first), so that the CPU can work on bands with small
 * buffers; the GPU passes whole images (first = 0).
 */

#ifndef CANNY_CORE_H
#define CANNY_CORE_H

#include <stddef.h>

#ifdef __CUDACC__
#	define CANNY_HD __host__ __device__
#else
#	define CANNY_HD
#endif

/* ceil(3 sigma) for sigma up to 10 */
#define CANNY_MAX_RADIUS 30

/* tan(22.5 deg), in units of 1/65536 */
#define CANNY_TAN 27146LL
#define CANNY_ONE 65536LL

enum {
	CANNY_NONE = 0,
	CANNY_WEAK = 1, /* edge only if connected to a strong one */
	CANNY_EDGE = 2
};

/* Symmetric mirroring (-1 -> 0, n -> n-1), as for NLM. */
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

/* Away from the border no index needs mirroring: same values, no modulo. */
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

/* Rounded to units of 1/256 gray level. */
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

/* dir may be NULL; sectors are described in the function. */
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
		 * 0: near horizontal (|gy|/|gx| <= tan 22.5), 2: near vertical,
		 * 1: diagonal with gx, gy of the same sign, 3: opposite signs
		 */
	if (ay * CANNY_ONE <= ax * CANNY_TAN)
		*dir = 0;
	else if (ay * CANNY_TAN >= ax * CANNY_ONE)
		*dir = 2;
	else
		*dir = ((gx > 0) == (gy > 0)) ? 1 : 3;
}

/*
 * Non-maximum suppression and thresholding. Strictly greater than one
 * neighbor and at least the other, so a two-pixel-wide ridge keeps one.
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
