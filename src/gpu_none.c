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
 *
 * @return IMG_ERR_UNSUPPORTED.
 */
int
gpu_denoise(NlmJob *job, GpuTiming *timing)
{
	(void)job;
	(void)timing;
	return IMG_ERR_UNSUPPORTED;
}
