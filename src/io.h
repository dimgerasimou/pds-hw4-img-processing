/**
 * @file io.h
 * @brief Input/output path resolution and the processing stages.
 *
 * <input> and -o <output> become (input, output) path pairs, following the
 * conventions of cp. The work is split into stages (read, decode, denoise,
 * edges, encode, write), each run in parallel over a range of images. The
 * caller runs them batch by batch and releases every finished batch, to
 * bound memory; stage wall times accumulate over the batches.
 */

#ifndef IO_H
#define IO_H

#include <stddef.h>

#include "canny.h"
#include "gpu.h"
#include "image.h"
#include "nlm.h"
#include "progress.h"

/* Stage not attempted (distinct from every IMG_* code). */
#define IO_NOT_DONE (-1)

enum {
	STAGE_READ = 0,
	STAGE_DECODE,
	STAGE_DENOISE,
	STAGE_EDGES,
	STAGE_ENCODE,
	STAGE_WRITE,
	STAGE_COUNT
};

typedef struct {
	int enabled;
	NlmParams params;
} DenoiseConfig;

typedef struct {
	int enabled;
	CannyParams params;
	CannySetup setup;
} EdgesConfig;

typedef struct {
	char *in_path;
	char *out_path;                /* NULL if not writing */
	int in_format;
	unsigned char *buf;            /* file contents, or encoder output */
	size_t buf_len;
	Image img;
	unsigned int width;            /* kept after the pixels are released */
	unsigned int height;
	double sigma;                  /* noise level used by denoising */
	double strength;               /* h / sigma used by denoising */
	size_t units;                  /* progress units of its denoising */
	NlmJob job;                    /* GPU: prepared on the CPU, run on the GPU */
	int err[STAGE_COUNT];          /* IMG_* or IO_NOT_DONE */
	int sys_errno[STAGE_COUNT];
	const char *detail;            /* decoder message on failure */
	double time_s[STAGE_COUNT];
	size_t bytes[STAGE_COUNT];
} ImageItem;

typedef struct {
	ImageItem *items;
	size_t count;
	int out_format;
	int quiet;                      /* do not report failures */
	DenoiseConfig denoise;
	EdgesConfig edges;
	GpuTiming gpu_time;
	int performed[STAGE_COUNT];
	double wall_time_s[STAGE_COUNT];/* summed over batches */
} ImageSet;

/**
 * @brief Builds the image set from the input and output arguments.
 *
 * Lists the images, picks the output format and paths, rejects name
 * collisions and creates the output directory. Without @p format, a single
 * output file's extension decides, otherwise PGM. Returns 1 on error
 * (already reported).
 */
int io_resolve(const char *input, const char *output, int format, ImageSet *set);

/**
 * @brief Runs one stage over the images [first, last).
 *
 * An image takes part if it passed the previous stage that ran. Denoising
 * and edge detection split the images into bands of rows spread over all
 * threads; the other stages are parallel across images. Every image
 * advances @p progress by its units, taking part or not, so the bar always
 * completes. Buffers are freed as soon as no later stage needs them.
 *
 * @return Number of images that failed.
 */
size_t io_run(ImageSet *set, int stage, size_t first, size_t last,
              Progress *progress);

/**
 * @brief GPU pipeline, CPU part: prepares the images of a range for
 *        denoising (noise estimate, padding, weight table).
 */
size_t io_prepare(ImageSet *set, size_t first, size_t last, Progress *progress);

/**
 * @brief GPU pipeline, GPU part: runs the filters on a range.
 *
 * Images whose denoising the GPU cannot do go to the CPU, with the same
 * result. Stage and per-image times are GPU times (CUDA events).
 */
size_t io_gpu_filter(ImageSet *set, size_t first, size_t last, Progress *progress);

/** @brief Frees the buffers of a range; results and dimensions are kept. */
void io_release(ImageSet *set, size_t first, size_t last);

/** @brief Clears all per-run state before running the stages again. */
void io_reset(ImageSet *set);

/**
 * @brief Progress units per image: 1 per I/O, codec and edge stage, and
 *        nlm_units() for denoising, which dominates the run time.
 */
size_t io_progress_units(const ImageSet *set, int write);

const char* io_stage_name(int stage);

void io_free(ImageSet *set);

#endif /* IO_H */
