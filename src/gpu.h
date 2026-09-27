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
 * The CPU prepares every image as usual (noise estimate, mirrored padding,
 * weight table; see nlm_job_prepare()), the GPU computes the denoised pixels
 * from the padded image and the table, and the CPU collects the result. The
 * GPU computes the same integer patch sums, uses the same weights and adds
 * them in the same order as the CPU, so the output is bit-for-bit identical.
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
 * With @p keep set, the result is left on the GPU for gpu_edges() instead
 * of being downloaded (the job's output buffer is then not filled).
 *
 * @param[in,out] job    Job prepared with nlm_job_prepare().
 * @param[in,out] timing GPU time of the steps, added to (may be NULL).
 * @param[in]     keep   Non-zero to keep the result on the GPU.
 *
 * @return IMG_OK, IMG_ERR_UNSUPPORTED if the job has no weight table (its
 *         parameters need more weights than the table limit; run it on the
 *         CPU instead), or IMG_ERR_GPU on a CUDA error (already reported).
 */
int gpu_denoise(NlmJob *job, GpuTiming *timing, int keep);

/**
 * @brief Detects the edges of an image on the GPU, replacing it with the
 *        edge map (255 on edges, 0 elsewhere).
 *
 * Uploads the image (or, with @p on_gpu set, uses the result gpu_denoise()
 * kept on the GPU), runs the blur, gradient/suppression and hysteresis
 * kernels, and downloads the edge map into the image's buffer. The result
 * is identical to the CPU's (canny_band() and canny_hysteresis()).
 *
 * @note Not thread-safe: call from one thread (the pipeline's GPU thread).
 *
 * @param[in,out] img    Image (its pixels are not read when @p on_gpu is set).
 * @param[in]     s      Setup from canny_setup().
 * @param[in,out] timing GPU time of the steps, added to (may be NULL).
 * @param[in]     on_gpu Non-zero to use the denoised image kept on the GPU.
 *
 * @return IMG_OK, or IMG_ERR_GPU on a CUDA error (already reported).
 */
int gpu_edges(Image *img, const CannySetup *s, GpuTiming *timing, int on_gpu);

#ifdef __cplusplus
}
#endif

#endif /* GPU_H */
