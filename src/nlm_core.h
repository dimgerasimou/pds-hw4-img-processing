/**
 * @file nlm_core.h
 * @brief The NLM weight function, compiled into both the C code and the
 *        CUDA kernel.
 *
 * exp(-x) from adds, multiplies and bit operations only, with no libm exp
 * and no contraction of separate multiplies and adds (see CFLAGS and
 * NVCCFLAGS), so that the CPU and the GPU compute the same float bit for bit.
 * The only fused operations are explicit fmaf() calls: correctly rounded by
 * definition, so identical on both sides, and one instruction on the GPU.
 */

#ifndef NLM_CORE_H
#define NLM_CORE_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef __CUDACC__
#	define NLM_HD __host__ __device__
#else
#	define NLM_HD
#endif

static inline NLM_HD uint32_t
nlm_float_bits(float f)
{
#ifdef __CUDA_ARCH__
	return __float_as_uint(f);
#else
	uint32_t u;

	memcpy(&u, &f, sizeof(u));
	return u;
#endif
}

static inline NLM_HD float
nlm_bits_float(uint32_t u)
{
#ifdef __CUDA_ARCH__
	return __uint_as_float(u);
#else
	float f;

	memcpy(&f, &u, sizeof(f));
	return f;
#endif
}

/*
 * exp(-max(ssd - offset, 0) * scale * ln 2), to about 1e-7 relative error.
 * Write the argument as t = n + g / ln 2 with n an integer: the weight is
 * 2^-n * e^u, u = g in [-ln2/2, ln2/2], from a Taylor series.
 */
static inline NLM_HD float
nlm_weight(int ssd, float offset, float scale)
{
	const float magic = 12582912.0f; /* 1.5 * 2^23: adding it rounds to an integer */
	float d = (float)ssd - offset;
	float t = (d > 0.0f ? d : 0.0f) * scale;
	float r, u, p;
	int n;

	if (t > 92.0f)
		t = 92.0f;

	r = t + magic;
	n = (int)(nlm_float_bits(r) - 0x4B400000u);
	u = ((r - magic) - t) * 0.693147181f; /* r - magic == n, and n - t is exact */

	p = 1.38888889e-3f;                    /* 1 / 720 */
	p = fmaf(p, u, 8.33333333e-3f);        /* 1 / 120 */
	p = fmaf(p, u, 4.16666667e-2f);        /* 1 / 24 */
	p = fmaf(p, u, 1.66666667e-1f);        /* 1 / 6 */
	p = fmaf(p, u, 0.5f);
	p = fmaf(p, u, 1.0f);
	p = fmaf(p, u, 1.0f);

	return p * nlm_bits_float((uint32_t)(127 - n) << 23);
}

#endif /* NLM_CORE_H */
