/**
 * @file gpu_none.c
 * @brief GPU backend for builds without CUDA: reports the GPU as unavailable.
 *
 * Linked instead of gpu.cu when no CUDA toolkit is found (see the Makefile),
 * so that the program builds and runs, CPU only, everywhere.
 */

#include "error.h"
#include "gpu.h"

/**
 * @brief Reports that this build has no GPU support.
 *
 * @param[out] info Unused.
 *
 * @return 1.
 */
int
gpu_init(GpuInfo *info)
{
	(void)info;
	uerrf("this build has no CUDA support (rebuild with a CUDA toolkit installed)");
	return 1;
}

/**
 * @brief Nothing to release.
 */
void
gpu_shutdown(void)
{
}

/**
 * @brief Not available in this build.
 *
 * @param[in,out] job    Unused.
 * @param[in,out] timing Unused.
 * @param[in]     keep   Unused.
 *
 * @return IMG_ERR_UNSUPPORTED.
 */
int
gpu_denoise(NlmJob *job, GpuTiming *timing, int keep)
{
	(void)job;
	(void)timing;
	(void)keep;
	return IMG_ERR_UNSUPPORTED;
}

/**
 * @brief Not available in this build.
 *
 * @param[in,out] img    Unused.
 * @param[in]     s      Unused.
 * @param[in,out] timing Unused.
 * @param[in]     on_gpu Unused.
 *
 * @return IMG_ERR_UNSUPPORTED.
 */
int
gpu_edges(Image *img, const CannySetup *s, GpuTiming *timing, int on_gpu)
{
	(void)img;
	(void)s;
	(void)timing;
	(void)on_gpu;
	return IMG_ERR_UNSUPPORTED;
}
