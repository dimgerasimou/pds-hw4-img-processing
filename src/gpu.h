/**
 * @file gpu.h
 * @brief GPU (CUDA) backend for NLM denoising and Canny edge detection.
 *
 * The CPU still prepares every image for denoising (see nlm_job_prepare()).
 * The GPU performs exactly the CPU's operations in the same order, so the
 * output is bit-for-bit identical.
 */

#ifndef GPU_H
#define GPU_H

#include "canny.h"
#include "nlm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	char name[128];
	int cc_major;
	int cc_minor;
	int sm_count;
	double mem_gb;
	int driver_version;
	int runtime_version;
} GpuInfo;

/* GPU-side times, measured with CUDA events. */
typedef struct {
	double upload_s;
	double denoise_s;
	double edges_s;       /* includes hysteresis_s */
	double hysteresis_s;
	double download_s;
	double launches;      /* hysteresis kernel launches */
	double tiles;         /* hysteresis tiles processed */
	double tiles_all;     /* tiles if every tile were active in every launch */
} GpuTiming;

/* Status of a task still waiting for the GPU (distinct from every IMG_* code). */
#define GPU_PENDING (-2)

typedef struct {
	NlmJob *job;      /* NULL: edge detection only */
	Image *img;       /* always set; receives the edge map */
	int status;       /* IMG_OK, IMG_ERR_UNSUPPORTED or IMG_ERR_GPU */
	double denoise_s;
	double edges_s;
	void *user;
} GpuTask;

/** @brief Initializes the first CUDA device. Returns 1 if there is none. */
int gpu_init(GpuInfo *info);

void gpu_shutdown(void);

/**
 * @brief Runs a batch of images through the GPU filters.
 *
 * Denoises the tasks that have a job, then detects edges if @p cs is given.
 * The edge map replaces img->data; with denoising only, the result goes to
 * job->out. Two images are in flight at a time, so that copies overlap with
 * kernels. Tasks the GPU cannot denoise get IMG_ERR_UNSUPPORTED, untouched.
 * @p done (may be NULL) is called as each task completes.
 *
 * Not thread-safe: call from one thread.
 *
 * @return Number of tasks that failed with IMG_ERR_GPU.
 */
size_t gpu_run(GpuTask *tasks, size_t n, const CannySetup *cs, GpuTiming *timing,
               void (*done)(GpuTask *t, void *ctx), void *ctx);

#ifdef __cplusplus
}
#endif

#endif /* GPU_H */
