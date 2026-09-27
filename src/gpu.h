/**
 * @file gpu.h
 * @brief GPU (CUDA) backend for NLM denoising and Canny edge detection.
 *
 * Declares the C interface to the CUDA code. Following the same split as the
 * rest of the project, all program logic stays in C; the CUDA translation
 * unit (gpu.cu) holds only the kernel and thin wrappers with C linkage.
 * Builds without CUDA link gpu_none.c instead, where the GPU is reported as
 * unavailable.
 *
 * The CPU prepares every image for denoising as usual (noise estimate,
 * mirrored padding, weight table; see nlm_job_prepare()), and the GPU
 * computes the denoised pixels and/or the edge map. The GPU performs exactly
 * the CPU's integer and floating-point operations, in the same order, so the
 * output is bit-for-bit identical.
 */

#ifndef GPU_H
#define GPU_H

#include "canny.h"
#include "nlm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------- */
/*                              Data Structures                              */
/* ------------------------------------------------------------------------- */

/**
 * @struct GpuInfo
 * @brief Information about the GPU in use.
 */
typedef struct {
	char name[128];      /**< Device name */
	int cc_major;        /**< Compute capability, major */
	int cc_minor;        /**< Compute capability, minor */
	int sm_count;        /**< Number of streaming multiprocessors */
	double mem_gb;       /**< Total device memory in GB */
	int driver_version;  /**< CUDA driver version (e.g. 12040 for 12.4) */
	int runtime_version; /**< CUDA runtime version */
} GpuInfo;

/**
 * @struct GpuTiming
 * @brief GPU time of the filters, split by step, in seconds.
 *
 * Measured with CUDA events, i.e. by the GPU itself.
 */
typedef struct {
	double upload_s;      /**< Host to device: images and weight tables */
	double denoise_s;     /**< NLM kernel */
	double edges_s;       /**< Canny kernels, hysteresis included */
	double hysteresis_s;  /**< Canny hysteresis kernels (part of edges_s) */
	double download_s;    /**< Device to host: results */
	double launches;      /**< Hysteresis kernel launches */
	double tiles;         /**< Hysteresis tiles processed (active tiles) */
	double tiles_all;     /**< Hysteresis tiles in all launches (launches x tiles) */
} GpuTiming;

/* Status of a task still waiting for the GPU (distinct from every IMG_* code) */
#define GPU_PENDING (-2)

/**
 * @struct GpuTask
 * @brief One image for gpu_run().
 */
typedef struct {
	NlmJob *job;      /**< Prepared denoising job, or NULL for edge detection alone */
	Image *img;       /**< The image (always set): edge input without a job; receives the edge map */
	int status;       /**< Result: IMG_OK, IMG_ERR_UNSUPPORTED or IMG_ERR_GPU */
	double denoise_s; /**< GPU time of the denoising part (with its copies) */
	double edges_s;   /**< GPU time of the edge detection part (with its copies) */
	void *user;       /**< Caller's data */
} GpuTask;

/* ------------------------------------------------------------------------- */
/*                            Public API Functions                           */
/* ------------------------------------------------------------------------- */

/**
 * @brief Initializes the first CUDA device.
 *
 * Makes the calling thread wait for the GPU by sleeping rather than spinning,
 * so that a thread waiting for the GPU leaves its core to the CPU stages.
 *
 * @param[out] info Device information (may be NULL).
 *
 * @return 0 on success, 1 if no usable GPU (already reported).
 */
int gpu_init(GpuInfo *info);

/**
 * @brief Releases the GPU buffers. Safe to call when not initialized.
 */
void gpu_shutdown(void);

/**
 * @brief Runs a batch of images through the GPU filters, overlapping the
 *        copies of neighboring images with the kernels.
 *
 * For every task: denoising if it has a prepared job (with noise to
 * remove), then edge detection if @p cs is given. Two images are in flight
 * at a time, in alternating slots: while image i's kernels run, image i+1
 * is uploaded and image i-1 downloaded. The results are identical to the
 * CPU's.
 *
 * Results: with edge detection, the edge map replaces the task's image
 * pixels (img->data); with denoising only, it fills the job's output
 * (job->out.data). A task whose denoising the GPU does not support (its
 * weight table exceeds the limit) gets IMG_ERR_UNSUPPORTED without any GPU
 * work, for the caller to process on the CPU.
 *
 * @note Not thread-safe: call from one thread (the pipeline's GPU thread).
 *
 * @param[in,out] tasks  Tasks; each one's status and times are set.
 * @param[in]     n      Number of tasks.
 * @param[in]     cs     Edge detection setup, or NULL for denoising only.
 * @param[in,out] timing GPU time of the steps, added to (may be NULL).
 * @param[in]     done   Called for each task when it completes (may be NULL).
 * @param[in]     ctx    Passed to @p done.
 *
 * @return Number of tasks that failed on the GPU (IMG_ERR_GPU).
 */
size_t gpu_run(GpuTask *tasks, size_t n, const CannySetup *cs, GpuTiming *timing,
               void (*done)(GpuTask *t, void *ctx), void *ctx);

#ifdef __cplusplus
}
#endif

#endif /* GPU_H */
