/**
 * @file io.h
 * @brief Input/output path resolution and the parallel processing stages.
 *
 * Turns the <input> argument and the optional -o <output> argument into a
 * list of (input path, output path) pairs, following the conventions of
 * cp(1):
 *
 *   input  output       behaviour
 *   -----  -----------  ----------------------------------------------------
 *   file   (none)       read only
 *   file   file         write to that file
 *   file   dir          write to dir/<input name>.<output ext>
 *   dir    (none)       read every image in dir
 *   dir    dir          write dir/<name>.<output ext> for every image
 *                       (dir is created)
 *   dir    file         error
 *
 * A directory input is scanned non-recursively for regular files with a
 * supported image extension, sorted by name. Writing over the input itself
 * is refused, and so are two inputs that would map to the same output name
 * (e.g. scan.png and scan.jpg both becoming scan.pgm).
 *
 * The work is split into stages, each run over all images in parallel:
 *
 *   read    file   -> memory   (pure I/O)
 *   decode  memory -> pixels   (pure CPU)
 *   denoise pixels -> pixels   (optional, NLM)
 *   edges   pixels -> pixels   (optional, Canny edge map)
 *   encode  pixels -> memory   (pure CPU)
 *   write   memory -> file     (pure I/O)
 *
 * so that I/O, (de)compression and filtering are timed separately. An image
 * that fails a stage is skipped by all following stages.
 *
 * The I/O and codec stages run in parallel across images. The denoise stage
 * splits every image of the batch into bands of rows and spreads all of
 * them over the threads, so that both many small images and a single large
 * one keep every thread busy.
 *
 * To bound memory, the caller runs the stages batch by batch: every stage
 * operates on a range [first, last) of the set, and io_release() frees a
 * finished batch. Stage wall times accumulate over all batches.
 */

#ifndef IO_H
#define IO_H

#include <stddef.h>

#include "canny.h"
#include "gpu.h"
#include "image.h"
#include "nlm.h"
#include "progress.h"

/** Marks a stage that was not attempted (distinct from every IMG_* code). */
#define IO_NOT_DONE (-1)

/* ------------------------------------------------------------------------- */
/*                              Stage Enumeration                            */
/* ------------------------------------------------------------------------- */

/**
 * @enum Stages
 * @brief Processing stages, in execution order.
 */
enum {
	STAGE_READ = 0, /**< Read file contents into memory */
	STAGE_DECODE,   /**< Decode memory into pixels */
	STAGE_DENOISE,  /**< NLM denoising of the pixels */
	STAGE_EDGES,    /**< Canny edge detection (replaces the pixels) */
	STAGE_ENCODE,   /**< Encode pixels into memory */
	STAGE_WRITE,    /**< Write memory to file */
	STAGE_COUNT     /**< Sentinel: number of stages */
};

/* ------------------------------------------------------------------------- */
/*                              Data Structures                              */
/* ------------------------------------------------------------------------- */

/**
 * @struct DenoiseConfig
 * @brief Configuration of the denoising stage.
 */
typedef struct {
	int enabled;      /**< Non-zero to run the stage */
	NlmParams params; /**< Filter parameters */
} DenoiseConfig;

/**
 * @struct EdgesConfig
 * @brief Configuration of the edge detection stage.
 */
typedef struct {
	int enabled;       /**< Non-zero to run the stage */
	CannyParams params; /**< Detector parameters */
	CannySetup setup;  /**< Values derived from the parameters */
} EdgesConfig;

/**
 * @struct ImageItem
 * @brief One image of the set, with its paths, data and per-stage status.
 */
typedef struct {
	char *in_path;                 /**< Path the image is read from */
	char *out_path;                /**< Path the image is written to, NULL if not writing */
	int in_format;                 /**< Detected input format (IMG_FMT_*) */
	unsigned char *buf;            /**< Encoded bytes: file contents or encoder output */
	size_t buf_len;                /**< Size of buf in bytes */
	Image img;                     /**< Decoded pixels */
	unsigned int width;            /**< Width after decode (kept after release) */
	unsigned int height;           /**< Height after decode (kept after release) */
	double sigma;                  /**< Noise standard deviation used by denoise */
	size_t units;                  /**< Progress units of the denoise stage (see io_run) */
	NlmJob job;                    /**< GPU denoising: prepared on the CPU, run on the GPU */
	int err[STAGE_COUNT];          /**< IMG_* result per stage, or IO_NOT_DONE */
	int sys_errno[STAGE_COUNT];    /**< errno per stage for IMG_ERR_SYS */
	const char *detail;            /**< Decoder message on decode failure, or NULL */
	double time_s[STAGE_COUNT];    /**< Time spent per stage in seconds */
	size_t bytes[STAGE_COUNT];     /**< Data volume per stage (see io_run) */
} ImageItem;

/**
 * @struct ImageSet
 * @brief All the images a run operates on, and per-stage totals.
 */
typedef struct {
	ImageItem *items;               /**< Array of images */
	size_t count;                   /**< Number of images */
	int out_format;                 /**< Output format (IMG_FMT_*) */
	int quiet;                      /**< Non-zero to not report failures */
	DenoiseConfig denoise;          /**< Denoising stage configuration */
	EdgesConfig edges;              /**< Edge detection stage configuration */
	GpuTiming gpu_time;             /**< GPU time of the run (GPU filters only) */
	int performed[STAGE_COUNT];     /**< 1 if the stage ran */
	double wall_time_s[STAGE_COUNT];/**< Elapsed time of each stage, summed over batches */
} ImageSet;

/* ------------------------------------------------------------------------- */
/*                            Public API Functions                           */
/* ------------------------------------------------------------------------- */

/**
 * @brief Builds the image set from the input and output arguments.
 *
 * Resolves the input (file or directory), lists the images, decides the
 * output format and where each image is written, checks for name
 * collisions, and creates the output directory if needed. No image data is
 * read.
 *
 * The output format is @p format if given; otherwise, for a single output
 * file with a writable extension (.pgm/.png), the format of that extension;
 * otherwise PGM.
 *
 * @param[in]  input  Input file or directory.
 * @param[in]  output Output file or directory, or NULL to not write.
 * @param[in]  format Requested output format, or IMG_FMT_UNKNOWN for automatic.
 * @param[out] set    Image set to fill (zeroed on failure).
 *
 * @return 0 on success, 1 on error (already reported).
 */
int io_resolve(const char *input, const char *output, int format, ImageSet *set);

/**
 * @brief Runs one stage over the eligible images of a range, in parallel.
 *
 * An image is eligible if it passed the previous stage that ran (and, for
 * encode and write, has an output path). The denoise stage runs only if
 * enabled in the set, spreading bands of all images over the threads; all
 * other stages run in parallel across images. Failures are reported after
 * the parallel region, in input order, unless set->quiet is set. The wall
 * time of the call is added to the stage's total in the set.
 *
 * Every image of the range advances @p progress by its units for the
 * stage, eligible or not, so a bar sized with io_progress_units() always
 * reaches 100%. The denoise stage advances it band by band.
 *
 * Data volume recorded per image: bytes read (read), pixel bytes produced
 * (decode), pixel bytes filtered (denoise), pixel bytes consumed (encode),
 * bytes written (write).
 *
 * Buffers are released as soon as they are no longer needed: file contents
 * after decode, pixels after encode, encoded data after write.
 *
 * @param[in,out] set      Image set.
 * @param[in]     stage    STAGE_* value.
 * @param[in]     first    First image of the range.
 * @param[in]     last     One past the last image of the range.
 * @param[in,out] progress Progress bar to advance (may be disabled).
 *
 * @return Number of images that failed this stage.
 */
size_t io_run(ImageSet *set, int stage, size_t first, size_t last,
              Progress *progress);

/**
 * @brief GPU denoising, CPU part: prepares the images of a range.
 *
 * Computes, for every image, the noise estimate, the mirrored padding and
 * the weight table (nlm_job_prepare()), in parallel over the images (or,
 * with fewer images than threads, one image at a time with all threads).
 * The prepared jobs are kept in the items for io_gpu_filter().
 *
 * @param[in,out] set      Image set (denoising must be enabled).
 * @param[in]     first    First image of the range.
 * @param[in]     last     One past the last image of the range.
 * @param[in,out] progress Progress bar (advanced for images that are skipped).
 *
 * @return Number of images that failed.
 */
size_t io_prepare(ImageSet *set, size_t first, size_t last, Progress *progress);

/**
 * @brief GPU filters: denoises and/or detects the edges of a range.
 *
 * Hands all images of the range to the GPU (denoising for the images
 * prepared by io_prepare(), then edge detection), overlapping the copies of
 * neighboring images with the kernels. An image whose denoising parameters
 * exceed the GPU's limits is processed on the CPU instead, with the same
 * result. Meant to run on its own thread while the CPU stages work on other
 * batches. Stage and per-image times are GPU times (CUDA events).
 *
 * @param[in,out] set      Image set.
 * @param[in]     first    First image of the range.
 * @param[in]     last     One past the last image of the range.
 * @param[in,out] progress Progress bar (advanced per image and filter).
 *
 * @return Number of stage failures.
 */
size_t io_gpu_filter(ImageSet *set, size_t first, size_t last, Progress *progress);

/**
 * @brief Frees the buffers of a range of images.
 *
 * Paths, per-stage results and dimensions are kept for benchmarking.
 *
 * @param[in,out] set   Image set.
 * @param[in]     first First image of the range.
 * @param[in]     last  One past the last image of the range.
 */
void io_release(ImageSet *set, size_t first, size_t last);

/**
 * @brief Clears all per-run state, keeping paths and the output format.
 *
 * Call before running the stages again on the same set (e.g. repeated
 * benchmark trials). Releases any remaining buffers.
 *
 * @param[in,out] set Image set.
 */
void io_reset(ImageSet *set);

/**
 * @brief Progress units of one image over the stages that will run.
 *
 * Every I/O and codec stage counts 1 unit per image, and denoising counts
 * nlm_units() (the number of search offsets, 440 with the defaults), which
 * reflects that it dominates the run time.
 *
 * @param[in] set   Image set (its denoise configuration is used).
 * @param[in] write Non-zero if the encode and write stages will run.
 *
 * @return Units per image.
 */
size_t io_progress_units(const ImageSet *set, int write);

/**
 * @brief Returns the short name of a stage ("read", "decode", ...).
 *
 * @param[in] stage STAGE_* value.
 *
 * @return Static name string.
 */
const char* io_stage_name(int stage);

/**
 * @brief Frees all paths and buffers of the set.
 *
 * Safe to call on a zeroed set.
 *
 * @param[in,out] set Image set.
 */
void io_free(ImageSet *set);

#endif /* IO_H */
