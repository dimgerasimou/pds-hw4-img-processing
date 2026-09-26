/**
 * @file main.c
 * @brief Entry point for the imgfilter program.
 *
 * This implementation was developed for the purposes of the class:
 * Parallel and Distributed Systems,
 * Department of Electrical and Computer Engineering,
 * Aristotle University of Thessaloniki.
 *
 * Reads an image or a directory of images, processes them, and optionally
 * writes the results. Every stage (read, decode, encode, write) is
 * parallelized across images with OpenMP. Timings can be written as JSON
 * for benchmarking.
 *
 * Usage: ./imgfilter [-o output] [-f format] [-t threads] [-b bench.json] [-p] <input>
 */

#include <omp.h>
#include <stddef.h>
#include <string.h>

#include "args.h"
#include "benchmark.h"
#include "error.h"
#include "image.h"
#include "io.h"

/* ------------------------------------------------------------------------- */
/*                              Default Values                               */
/* ------------------------------------------------------------------------- */

#define DEFAULT_OUTPUT     NULL /* nothing is written */
#define DEFAULT_BENCH_PATH NULL /* no benchmark output */
#define DEFAULT_FORMAT     IMG_FMT_UNKNOWN /* automatic, see io_resolve() */
#define DEFAULT_PROGRESS   0

/* ------------------------------------------------------------------------- */
/*                                Main Function                              */
/* ------------------------------------------------------------------------- */

/**
 * @brief Program entry point.
 *
 * Parses command-line arguments, resolves the input and output paths,
 * loads and decodes the images, encodes and writes them back if requested,
 * and records timings.
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
	size_t failed = 0;
	int ret = 1;

	/* Command-line arguments with defaults */
	Args args = {
		.input      = NULL,
		.output     = DEFAULT_OUTPUT,
		.bench_path = DEFAULT_BENCH_PATH,
		.format     = DEFAULT_FORMAT,
		.threads    = (unsigned int)omp_get_max_threads(),
		.progress   = DEFAULT_PROGRESS,
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

	/* Initialize benchmark structure */
	if (args.bench_path) {
		bench = benchmark_init(args.input, args.output, args.threads);
		if (!bench)
			goto cleanup;
	}

	/* Load images: file -> memory -> pixels */
	failed += io_run(&set, STAGE_READ, args.progress);
	failed += io_run(&set, STAGE_DECODE, args.progress);

	/* TODO: filters (NLM denoising, Canny edge detection) run here */

	/* Save images: pixels -> memory -> file */
	if (args.output) {
		failed += io_run(&set, STAGE_ENCODE, args.progress);
		failed += io_run(&set, STAGE_WRITE, args.progress);
	}

	/* Write benchmark results in JSON format */
	if (bench) {
		if (benchmark_collect(bench, &set))
			goto cleanup;
		if (benchmark_write(bench, args.bench_path))
			goto cleanup;
	}

	/* Success only if every image made it through */
	ret = failed ? 1 : 0;

cleanup:
	benchmark_free(bench);
	io_free(&set);
	return ret;
}
