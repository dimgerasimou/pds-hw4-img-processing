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
 * @brief Not available in this build: every task is unsupported.
 *
 * @param[in,out] tasks  Tasks (status set to IMG_ERR_UNSUPPORTED).
 * @param[in]     n      Number of tasks.
 * @param[in]     cs     Unused.
 * @param[in,out] timing Unused.
 * @param[in]     done   Called for each task (may be NULL).
 * @param[in]     ctx    Passed to @p done.
 *
 * @return 0.
 */
size_t
gpu_run(GpuTask *tasks, size_t n, const CannySetup *cs, GpuTiming *timing,
        void (*done)(GpuTask *t, void *ctx), void *ctx)
{
	(void)cs;
	(void)timing;

	for (size_t i = 0; i < n; i++) {
		tasks[i].status = IMG_ERR_UNSUPPORTED;
		if (done)
			done(&tasks[i], ctx);
	}

	return 0;
}
