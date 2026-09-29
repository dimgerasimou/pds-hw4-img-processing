/**
 * @file benchmark.h
 * @brief Timing statistics of the pipeline stages, written as JSON.
 */

#ifndef BENCHMARK_H
#define BENCHMARK_H

#ifndef PATH_MAX
#	include <linux/limits.h>
#endif
#ifndef PATH_MAX
#	define PATH_MAX 4096
#endif

#include <stddef.h>

#include "gpu.h"
#include "io.h"

typedef struct {
	double mean_time_s;
	double std_dev_s;
	double median_time_s;
	double min_time_s;
	double max_time_s;
	double total_time_s;
} Statistics;

typedef struct {
	int performed;
	size_t images;
	size_t failed;
	double data_mib;
	double throughput_mib_s;
	double images_per_sec;
	Statistics wall_time;
	Statistics per_image;
} StageResult;

typedef struct {
	char cpu_info[128];
	unsigned int cpu_cores;
	double ram_gb;
	double swap_gb;
	int has_gpu;
	GpuInfo gpu;
} SystemInfo;

typedef struct {
	DenoiseConfig config;
	int gpu;
	double sigma_mean;
	double sigma_min;
	double sigma_max;
	double strength_mean;
	double strength_min;
	double strength_max;
} DenoiseInfo;

enum {
	GPU_UPLOAD = 0,
	GPU_DENOISE,
	GPU_EDGES,
	GPU_HYSTERESIS,
	GPU_DOWNLOAD,
	GPU_LAUNCHES,
	GPU_TILES,
	GPU_TILES_ALL,
	GPU_METRICS
};

typedef struct {
	char timestamp[32];
	unsigned int threads;
	unsigned int trials;
	unsigned int wtrials;
	size_t batch_size;
	size_t batches;
} BenchmarkInfo;

typedef struct {
	char input_path[PATH_MAX];
	char output_path[PATH_MAX];
	int output_format;
	size_t images;
	size_t formats[IMG_FMT_COUNT];
	unsigned long long pixels;
	unsigned int min_width;
	unsigned int min_height;
	unsigned int max_width;
	unsigned int max_height;
} DatasetInfo;

typedef struct {
	double cpu_peak_rss_gb;
	long major_page_faults;
	long minor_page_faults;
} MemoryInfo;

typedef struct {
	SystemInfo sys_info;
	BenchmarkInfo benchmark_info;
	DatasetInfo dataset_info;
	DenoiseInfo denoise_info;
	EdgesConfig edges_info;
	StageResult results[STAGE_COUNT];
	Statistics pipeline_time;
	Statistics gpu_time[GPU_METRICS];
	MemoryInfo memory;

	unsigned int trials_done;
	size_t images;
	double *wall[STAGE_COUNT];
	double *samples[STAGE_COUNT];
	size_t nsamples[STAGE_COUNT];
	double *pipeline;
	double *gpu[GPU_METRICS];
	double trial_start;
	long majflt_start;
	long minflt_start;
} Benchmark;

double now_sec(void);

Benchmark* benchmark_init(const char *input, const char *output,
                          const unsigned int threads, const unsigned int trials,
                          const unsigned int wtrials, const size_t batch,
                          const size_t batches, const size_t images,
                          const DenoiseConfig *denoise, const EdgesConfig *edges);

void benchmark_set_gpu(Benchmark *b, const GpuInfo *info);

void benchmark_free(Benchmark *b);

void benchmark_trial_start(Benchmark *b);

/**
 * @brief Records a finished trial: stage and per-image times from @p set,
 *        pipeline time and page faults. The first trial also records the
 *        dataset information.
 */
int benchmark_trial_end(Benchmark *b, const ImageSet *set);

/** @brief Summarizes the trials and writes the JSON to @p path ("-": stdout). */
int benchmark_write(Benchmark *b, const char *path);

#endif /* BENCHMARK_H */
