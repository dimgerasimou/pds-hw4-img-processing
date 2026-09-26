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
 * Non-Local Means, and optionally writes the results. Images are processed
 * in batches to bound memory use; within a batch, every stage is
 * parallelized with OpenMP, across images or, for the filters, optionally
 * within each image. Timings can be written as JSON for
 * benchmarking.
 *
 * Usage: ./imgfilter [-o output] [-f format] [-t threads] [-B batch]
 *                    [-b bench.json [-n trials] [-w wtrials]] [-p]
 *                    [-d [-P patch] [-S search] [-H k] [-N sigma] [-A method]
 *                        [-m mode]] <input>
 */

#include <omp.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "args.h"
#include "benchmark.h"
#include "error.h"
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
#define DEFAULT_NLM_METHOD NLM_INTEGRAL
#define DEFAULT_MODE       PAR_AUTO

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
 * @param[in,out] progress Progress bar, one step per image and stage.
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

		/* TODO: Canny edge detection runs here */

		/* Save images: pixels -> memory -> file */
		if (write) {
			failed += io_run(set, STAGE_ENCODE, first, last, progress);
			failed += io_run(set, STAGE_WRITE, first, last, progress);
		}

		io_release(set, first, last);
	}

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
	Benchmark *bench = NULL;
	Progress progress;
	size_t failed = 0, batch, batches;
	unsigned int runs;
	int stages, ret = 1;

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
		.mode       = DEFAULT_MODE,
		.nlm        = {
			.patch    = DEFAULT_NLM_PATCH,
			.search   = DEFAULT_NLM_SEARCH,
			.h_factor = DEFAULT_NLM_H,
			.sigma    = DEFAULT_NLM_SIGMA,
			.method   = DEFAULT_NLM_METHOD,
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

	/* Resolve input/output paths, output format, and list the images */
	if (io_resolve(args.input, args.output, args.format, &set))
		goto cleanup;

	/* Filter configuration */
	set.denoise.enabled = args.denoise;
	set.denoise.mode = args.mode;
	set.denoise.params = args.nlm;

	/* Batch size: 0 means the whole set at once */
	batch = (args.batch == 0 || args.batch > set.count) ? set.count : args.batch;
	batches = (set.count + batch - 1) / batch;

	/* Initialize benchmark structure */
	if (args.bench_path) {
		bench = benchmark_init(args.input, args.output, args.threads, args.trials,
		                       args.wtrials, batch, batches, set.count, &set.denoise);
		if (!bench)
			goto cleanup;
	}

	/* One progress step per image and stage */
	stages = 2 + (args.denoise ? 1 : 0) + (args.output ? 2 : 0);
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

		progress_init(&progress, label, set.count * (size_t)stages,
		              (size_t)stages, args.progress);
		progress_set(&progress, NULL, note);

		if (bench && timed)
			benchmark_trial_start(bench);

		run_failed = run_pipeline(&set, batch, args.output != NULL, &progress);

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
	benchmark_free(bench);
	io_free(&set);
	return ret;
}
