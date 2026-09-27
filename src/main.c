/**
 * @file main.c
 * @brief Entry point for the imgfilter program.
 *
 * This implementation was developed for the purposes of the class:
 * Parallel and Distributed Systems,
 * Department of Electrical and Computer Engineering,
 * Aristotle University of Thessaloniki.
 *
 * Reads an image or a directory of images, optionally denoises them with
 * Non-Local Means and/or detects their edges with Canny, and optionally
 * writes the results. Images are processed
 * in batches to bound memory use; within a batch, every stage is
 * parallelized with OpenMP, across images, and denoising also within them.
 * With -g, the filters run on the GPU (CUDA), pipelined with the CPU stages. Timings can be written as JSON for
 * benchmarking.
 *
 * Usage: ./imgfilter [-o output] [-f format] [-t threads] [-B batch]
 *                    [-b bench.json [-n trials] [-w wtrials]] [-p]
 *                    [-d [-P patch] [-S search] [-H k] [-N sigma]]
 *                    [-e [-G sigma] [-l low] [-u high]] [-g] <input>
 */

#include <omp.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "args.h"
#include "benchmark.h"
#include "error.h"
#include "gpu.h"
#include "image.h"
#include "io.h"
#include "progress.h"

/* ------------------------------------------------------------------------- */
/*                              Default Values                               */
/* ------------------------------------------------------------------------- */

#define DEFAULT_OUTPUT     NULL /* nothing is written */
#define DEFAULT_BENCH_PATH NULL /* no benchmark output */
#define DEFAULT_FORMAT     IMG_FMT_UNKNOWN /* automatic, see io_resolve() */
#define DEFAULT_PROGRESS   0
#define DEFAULT_BATCH      256  /* images per batch; bounds peak memory */
#define DEFAULT_TRIALS     1    /* timed benchmark trials */
#define DEFAULT_WTRIALS    0    /* warmup benchmark trials */

/*
 * NLM defaults: Buades et al. (IPOL 2011) recommend 5x5 patches, a 21x21
 * search window and h = 0.4 sigma for moderate noise (sigma 15-30).
 */
#define DEFAULT_NLM_PATCH  2    /* 5x5 patches */
#define DEFAULT_NLM_SEARCH 10   /* 21x21 search window */
#define DEFAULT_NLM_H      0.4  /* h = 0.4 * sigma */
#define DEFAULT_NLM_SIGMA  -1.0 /* estimate per image */

/* Canny defaults: sigma 1.4 is the classic choice; thresholds in gradient units */
#define DEFAULT_CANNY_SIGMA 1.4
#define DEFAULT_CANNY_LOW   20.0
#define DEFAULT_CANNY_HIGH  50.0

/* ------------------------------------------------------------------------- */
/*                            Static Helper Functions                        */
/* ------------------------------------------------------------------------- */

/**
 * @brief Runs the whole pipeline once over the image set, batch by batch.
 *
 * Each batch goes through every stage, each stage parallel over the batch,
 * and is then released, so that at most @p batch images are in memory.
 *
 * @param[in,out] set      Image set (reset by the caller).
 * @param[in]     batch    Images per batch.
 * @param[in]     write    Non-zero to encode and write the results.
 * @param[in,out] progress Progress bar (see io_progress_units()).
 *
 * @return Number of stage failures.
 */
static size_t
run_pipeline(ImageSet *set, size_t batch, int write, Progress *progress)
{
	size_t failed = 0;

	for (size_t first = 0; first < set->count; first += batch) {
		size_t last = first + batch;

		if (last > set->count)
			last = set->count;

		/* Load images: file -> memory -> pixels */
		failed += io_run(set, STAGE_READ, first, last, progress);
		failed += io_run(set, STAGE_DECODE, first, last, progress);

		/* Filters: pixels -> pixels */
		if (set->denoise.enabled)
			failed += io_run(set, STAGE_DENOISE, first, last, progress);

		if (set->edges.enabled)
			failed += io_run(set, STAGE_EDGES, first, last, progress);

		/* Save images: pixels -> memory -> file */
		if (write) {
			failed += io_run(set, STAGE_ENCODE, first, last, progress);
			failed += io_run(set, STAGE_WRITE, first, last, progress);
		}

		io_release(set, first, last);
	}

	return failed;
}

/**
 * @brief Batch [first, last) of batch number @p b.
 */
static void
batch_range(const ImageSet *set, size_t batch, size_t b, size_t *first, size_t *last)
{
	*first = b * batch;
	*last = (*first + batch < set->count) ? *first + batch : set->count;
}

/**
 * @brief GPU pipeline, CPU part: loads (and prepares for denoising) batch @p b.
 */
static size_t
gpu_load(ImageSet *set, size_t batch, size_t b, Progress *progress)
{
	size_t first, last, failed = 0;

	batch_range(set, batch, b, &first, &last);
	failed += io_run(set, STAGE_READ, first, last, progress);
	failed += io_run(set, STAGE_DECODE, first, last, progress);
	if (set->denoise.enabled)
		failed += io_prepare(set, first, last, progress);
	return failed;
}

/**
 * @brief GPU pipeline, CPU part: saves and releases batch @p b.
 */
static size_t
gpu_store(ImageSet *set, size_t batch, size_t b, int write, Progress *progress)
{
	size_t first, last, failed = 0;

	batch_range(set, batch, b, &first, &last);
	if (write) {
		failed += io_run(set, STAGE_ENCODE, first, last, progress);
		failed += io_run(set, STAGE_WRITE, first, last, progress);
	}
	io_release(set, first, last);
	return failed;
}

/**
 * @brief Runs the pipeline once with the filters on the GPU.
 *
 * Three batches are in flight at a time: while the GPU filters batch k,
 * the CPU loads and prepares batch k+1 and encodes and writes batch k-1.
 * Two OpenMP sections run side by side; the GPU one is a single thread that
 * sleeps while waiting for the GPU, the CPU one uses all threads in its own
 * (nested) parallel regions.
 *
 * @param[in,out] set      Image set (reset by the caller).
 * @param[in]     batch    Images per batch.
 * @param[in]     write    Non-zero to encode and write the results.
 * @param[in,out] progress Progress bar (see io_progress_units()).
 *
 * @return Number of stage failures.
 */
static size_t
run_pipeline_gpu(ImageSet *set, size_t batch, int write, Progress *progress)
{
	const size_t nb = (set->count + batch - 1) / batch;
	size_t failed;

	failed = gpu_load(set, batch, 0, progress);

	for (size_t b = 0; b < nb; b++) {
		size_t f_gpu = 0, f_cpu = 0;

		#pragma omp parallel sections num_threads(2)
		{
			#pragma omp section
			{
				size_t first, last;

				batch_range(set, batch, b, &first, &last);
				f_gpu = io_gpu_filter(set, first, last, progress);
			}

			#pragma omp section
			{
				if (b + 1 < nb)
					f_cpu += gpu_load(set, batch, b + 1, progress);
				if (b > 0)
					f_cpu += gpu_store(set, batch, b - 1, write, progress);
			}
		}

		failed += f_gpu + f_cpu;
	}

	failed += gpu_store(set, batch, nb - 1, write, progress);
	return failed;
}

/* ------------------------------------------------------------------------- */
/*                                Main Function                              */
/* ------------------------------------------------------------------------- */

/**
 * @brief Program entry point.
 *
 * Parses command-line arguments, resolves the input and output paths, and
 * runs the pipeline. When benchmarking, the pipeline runs wtrials times
 * unrecorded, then trials times recorded, and the statistics are written
 * as JSON.
 *
 * @param[in] argc Argument count.
 * @param[in] argv Argument vector.
 *
 * @return 0 on success, 1 on error or if any image failed.
 */
int
main(int argc, char *argv[])
{
	ImageSet set;
	GpuInfo gpu_info;
	int gpu_ready = 0;
	Benchmark *bench = NULL;
	Progress progress;
	size_t failed = 0, batch, batches;
	unsigned int runs;
	size_t units;
	int ret = 1;

	/* Command-line arguments with defaults */
	Args args = {
		.input      = NULL,
		.output     = DEFAULT_OUTPUT,
		.bench_path = DEFAULT_BENCH_PATH,
		.format     = DEFAULT_FORMAT,
		.threads    = (unsigned int)omp_get_max_threads(),
		.batch      = DEFAULT_BATCH,
		.trials     = DEFAULT_TRIALS,
		.wtrials    = DEFAULT_WTRIALS,
		.progress   = DEFAULT_PROGRESS,
		.denoise    = 0,
		.gpu        = 0,
		.edges      = 0,
		.canny      = {
			.sigma = DEFAULT_CANNY_SIGMA,
			.low   = DEFAULT_CANNY_LOW,
			.high  = DEFAULT_CANNY_HIGH,
		},
		.nlm        = {
			.patch    = DEFAULT_NLM_PATCH,
			.search   = DEFAULT_NLM_SEARCH,
			.h_factor = DEFAULT_NLM_H,
			.sigma    = DEFAULT_NLM_SIGMA,
		},
	};

	memset(&set, 0, sizeof(set));

	/* Initialize error reporting with program name */
	err_init(argv[0]);

	/* Parse command-line arguments */
	switch (parse_args(argc, argv, &args)) {
	case 1:
		/* Parse error */
		return 1;

	case -1:
		/* Help requested */
		return 0;

	default:
		/* Success, continue */
		break;
	}

	omp_set_num_threads((int)args.threads);

	/* GPU pipeline: the CPU section runs its own parallel regions */
	if (args.gpu) {
		if (gpu_init(&gpu_info))
			goto cleanup;
		gpu_ready = 1;
		omp_set_max_active_levels(2);
	}

	/* Resolve input/output paths, output format, and list the images */
	if (io_resolve(args.input, args.output, args.format, &set))
		goto cleanup;

	/* Filter configuration */
	set.denoise.enabled = args.denoise;
	set.denoise.params = args.nlm;
	set.edges.enabled = args.edges;
	set.edges.params = args.canny;
	canny_setup(&args.canny, &set.edges.setup);

	/* Batch size: 0 means the whole set at once */
	batch = (args.batch == 0 || args.batch > set.count) ? set.count : args.batch;
	batches = (set.count + batch - 1) / batch;

	/* Initialize benchmark structure */
	if (args.bench_path) {
		bench = benchmark_init(args.input, args.output, args.threads, args.trials,
		                       args.wtrials, batch, batches, set.count, &set.denoise,
		                       &set.edges);
		if (!bench)
			goto cleanup;
		if (gpu_ready)
			benchmark_set_gpu(bench, &gpu_info);
	}

	/* Progress units per image, weighted by the cost of the stages */
	units = io_progress_units(&set, args.output != NULL);
	runs = args.wtrials + args.trials;

	for (unsigned int r = 0; r < runs; r++) {
		int timed = (r >= args.wtrials);
		const char *label = "run";
		char note[32] = "";
		size_t run_failed;

		if (runs > 1) {
			label = timed ? "trial" : "warmup";
			snprintf(note, sizeof(note), "%u/%u",
			         timed ? r - args.wtrials + 1 : r + 1,
			         timed ? args.trials : args.wtrials);
		}

		io_reset(&set);
		set.quiet = (r > 0); /* failures repeat every run: report them once */

		progress_init(&progress, label, set.count * units, units, args.progress);
		progress_set(&progress, NULL, note);

		if (bench && timed)
			benchmark_trial_start(bench);

		run_failed = args.gpu
			? run_pipeline_gpu(&set, batch, args.output != NULL, &progress)
			: run_pipeline(&set, batch, args.output != NULL, &progress);

		if (bench && timed && benchmark_trial_end(bench, &set))
			goto cleanup;

		progress_finish(&progress);

		if (r == 0)
			failed = run_failed;
	}

	/* Write benchmark results in JSON format */
	if (bench && benchmark_write(bench, args.bench_path))
		goto cleanup;

	/* Success only if every image made it through */
	ret = failed ? 1 : 0;

cleanup:
	if (gpu_ready)
		gpu_shutdown();
	benchmark_free(bench);
	io_free(&set);
	return ret;
}
