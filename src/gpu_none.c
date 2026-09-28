/**
 * @file gpu_none.c
 * @brief GPU backend for builds without CUDA.
 */

#include "error.h"
#include "gpu.h"

int
gpu_init(GpuInfo *info)
{
	(void)info;
	uerrf("this build has no CUDA support (rebuild with a CUDA toolkit installed)");
	return 1;
}

void
gpu_shutdown(void)
{
}

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
