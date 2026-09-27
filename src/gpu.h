/**
 * @file gpu.h
 * @brief GPU (CUDA) backend for NLM denoising.
 *
 * Declares the C interface to the CUDA code. Following the same split as the
 * rest of the project, all program logic stays in C; the CUDA translation
 * unit (gpu.cu) holds only the kernel and thin wrappers with C linkage.
 * Builds without CUDA link gpu_none.c instead, where the GPU is reported as
 * unavailable.
 *
 * The CPU prepares every image as usual (noise estimate, mirrored padding,
 * weight table; see nlm_job_prepare()), the GPU computes the denoised pixels
 * from the padded image and the table, and the CPU collects the result. The
 * GPU computes the same integer patch sums, uses the same weights and adds
 * them in the same order as the CPU, so the output is bit-for-bit identical.
 */

#ifndef GPU_H
#define GPU_H

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
 * @brief GPU time spent on denoising, split by step, in seconds.
 *
 * Measured with CUDA events, i.e. by the GPU itself.
 */
typedef struct {
	double upload_s;   /**< Host to device: padded image and weight table */
	double kernel_s;   /**< The NLM kernel */
	double download_s; /**< Device to host: denoised image */
} GpuTiming;

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
 * @brief Denoises a prepared job on the GPU, writing its output image.
 *
 * Uploads the padded image and the weight table, runs the kernel and
 * downloads the result into the job's output. Device buffers are reused
 * between calls and grown when needed.
 *
 * @note Not thread-safe: call from one thread (the pipeline's GPU thread).
 *
 * @param[in,out] job    Job prepared with nlm_job_prepare().
 * @param[in,out] timing GPU time of the three steps, added to (may be NULL).
 *
 * @return IMG_OK, IMG_ERR_UNSUPPORTED if the job has no weight table (its
 *         parameters need more weights than the table limit; run it on the
 *         CPU instead), or IMG_ERR_GPU on a CUDA error (already reported).
 */
int gpu_denoise(NlmJob *job, GpuTiming *timing);

#ifdef __cplusplus
}
#endif

#endif /* GPU_H */
