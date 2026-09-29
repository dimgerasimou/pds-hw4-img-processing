/**
 * @file main.c
 * @brief imgfilter: parallel NLM denoising and Canny edge detection.
 *
 * Parallel and Distributed Systems, ECE AUTh.
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

#define DEFAULT_OUTPUT     NULL
#define DEFAULT_BENCH_PATH NULL
#define DEFAULT_FORMAT     IMG_FMT_UNKNOWN /* see io_resolve() */
#define DEFAULT_PROGRESS   0
#define DEFAULT_BATCH      256  /* bounds peak memory */
#define DEFAULT_TRIALS     1
#define DEFAULT_WTRIALS    0

/* Buades et al. (IPOL 2011), for moderate noise */
#define DEFAULT_NLM_PATCH  2
#define DEFAULT_NLM_SEARCH 10
#define DEFAULT_NLM_H      0.0   /* automatic */
#define DEFAULT_NLM_SIGMA  -1.0 /* estimate per image */

#define DEFAULT_CANNY_SIGMA 1.4
#define DEFAULT_CANNY_LOW   20.0
#define DEFAULT_CANNY_HIGH  50.0

static size_t
run_pipeline(ImageSet *set, size_t batch, int write, Progress *progress)
{
	size_t failed = 0;

	for (size_t first = 0; first < set->count; first += batch) {
		size_t last = first + batch;

		if (last > set->count)
			last = set->count;

		failed += io_run(set, STAGE_READ, first, last, progress);
		failed += io_run(set, STAGE_DECODE, first, last, progress);

		if (set->denoise.enabled)
			failed += io_run(set, STAGE_DENOISE, first, last, progress);

		if (set->edges.enabled)
			failed += io_run(set, STAGE_EDGES, first, last, progress);

		if (write) {
			failed += io_run(set, STAGE_ENCODE, first, last, progress);
			failed += io_run(set, STAGE_WRITE, first, last, progress);
		}

		io_release(set, first, last);
	}

	return failed;
}

static void
batch_range(const ImageSet *set, size_t batch, size_t b, size_t *first, size_t *last)
{
	*first = b * batch;
	*last = (*first + batch < set->count) ? *first + batch : set->count;
}

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

/*
 * Three batches in flight: while the GPU filters batch b, the CPU loads
 * batch b+1 and saves batch b-1. The GPU section is one thread that sleeps
 * while waiting; the CPU section runs its own nested parallel regions.
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
			.sigma_scale = 1.0,
		},
	};

	memset(&set, 0, sizeof(set));

	err_init(argv[0]);

	switch (parse_args(argc, argv, &args)) {
	case 1:
		return 1;

	case -1:
		return 0;

	default:
		break;
	}

	omp_set_num_threads((int)args.threads);

	if (args.gpu) {
		if (gpu_init(&gpu_info))
			goto cleanup;
		gpu_ready = 1;
		omp_set_max_active_levels(2);
	}

	if (io_resolve(args.input, args.output, args.format, &set))
		goto cleanup;

	set.denoise.enabled = args.denoise;
	set.denoise.params = args.nlm;
	set.edges.enabled = args.edges;
	set.edges.params = args.canny;
	canny_setup(&args.canny, &set.edges.setup);

	/* 0: the whole set at once */
	batch = (args.batch == 0 || args.batch > set.count) ? set.count : args.batch;
	batches = (set.count + batch - 1) / batch;

	if (args.bench_path) {
		bench = benchmark_init(args.input, args.output, args.threads, args.trials,
		                       args.wtrials, batch, batches, set.count, &set.denoise,
		                       &set.edges);
		if (!bench)
			goto cleanup;
		if (gpu_ready)
			benchmark_set_gpu(bench, &gpu_info);
	}

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

	if (bench && benchmark_write(bench, args.bench_path))
		goto cleanup;

	ret = failed ? 1 : 0;

cleanup:
	if (gpu_ready)
		gpu_shutdown();
	benchmark_free(bench);
	io_free(&set);
	return ret;
}
