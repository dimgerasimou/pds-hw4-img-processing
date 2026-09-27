/**
 * @file benchmark.h
 * @brief Benchmarking framework for the image processing pipeline.
 *
 * Provides structures and functions to collect timing statistics for every
 * stage of a run (see io.h: read, decode, encode, write), capture system
 * information, and write everything as JSON for post-processing.
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

/* ------------------------------------------------------------------------- */
/*                          Benchmark Data Structures                        */
/* ------------------------------------------------------------------------- */

/**
 * @struct Statistics
 * @brief Statistical summary of a set of timings.
 */
typedef struct {
	double mean_time_s;   /**< Mean time in seconds */
	double std_dev_s;     /**< Standard deviation in seconds */
	double median_time_s; /**< Median time in seconds */
	double min_time_s;    /**< Minimum time in seconds */
	double max_time_s;    /**< Maximum time in seconds */
	double total_time_s;  /**< Sum of all times in seconds */
} Statistics;

/**
 * @struct StageResult
 * @brief Result of one stage over the whole image set.
 *
 * Counts and data volume are per trial (identical across trials). data_mib
 * is the data volume of the stage as defined by io_run(): bytes read,
 * pixels produced, pixels consumed, or bytes written.
 *
 * wall_time summarizes the elapsed time of the whole (parallel) stage over
 * the timed trials; throughput and rate use its median. per_image
 * summarizes the individual image times over all timed trials, so
 * per_image.total_time_s / (trials * median wall time) is the effective
 * parallelism achieved.
 */
typedef struct {
	int performed;            /**< 1 if the stage ran, else 0 */
	size_t images;            /**< Images processed successfully per trial */
	size_t failed;            /**< Images that failed per trial */
	double data_mib;          /**< Data volume per trial in MiB */
	double throughput_mib_s;  /**< data_mib / median wall time */
	double images_per_sec;    /**< images / median wall time */
	Statistics wall_time;     /**< Stage wall time over trials */
	Statistics per_image;     /**< Per-image time over all timed trials */
} StageResult;

/**
 * @struct SystemInfo
 * @brief System information captured during benchmark execution.
 */
typedef struct {
	char cpu_info[128];     /**< CPU model */
	unsigned int cpu_cores; /**< Logical CPUs online */
	double ram_gb;          /**< Total system RAM in gigabytes */
	double swap_gb;         /**< Total swap space in gigabytes */
	int has_gpu;            /**< Non-zero if the GPU was used */
	GpuInfo gpu;            /**< GPU information (if has_gpu) */
} SystemInfo;

/**
 * @struct DenoiseInfo
 * @brief Configuration of the denoising stage and the noise levels used.
 */
typedef struct {
	DenoiseConfig config; /**< Stage configuration */
	int gpu;              /**< Non-zero if denoising ran on the GPU */
	double sigma_mean;    /**< Mean noise standard deviation used */
	double sigma_min;     /**< Smallest noise standard deviation used */
	double sigma_max;     /**< Largest noise standard deviation used */
} DenoiseInfo;

/*
 * GPU metrics recorded per trial: the fields of GpuTiming, in this order
 * (the last one is a count of hysteresis launches, not a time).
 */
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

/**
 * @struct BenchmarkInfo
 * @brief Benchmark execution parameters.
 */
typedef struct {
	char timestamp[32];   /**< ISO 8601 timestamp of the run */
	unsigned int threads; /**< Number of OpenMP threads used */
	unsigned int trials;  /**< Number of timed trials */
	unsigned int wtrials; /**< Number of warmup trials (not recorded) */
	size_t batch_size;    /**< Images per batch */
	size_t batches;       /**< Number of batches per trial */
} BenchmarkInfo;

/**
 * @struct DatasetInfo
 * @brief Information about the processed image set.
 */
typedef struct {
	char input_path[PATH_MAX];     /**< Input file or directory */
	char output_path[PATH_MAX];    /**< Output file or directory ("" if none) */
	int output_format;             /**< Output format (IMG_FMT_*, UNKNOWN if none) */
	size_t images;                 /**< Number of images in the set */
	size_t formats[IMG_FMT_COUNT]; /**< Decoded images per input format */
	unsigned long long pixels;     /**< Total pixels of all loaded images */
	unsigned int min_width;        /**< Smallest image width */
	unsigned int min_height;       /**< Smallest image height */
	unsigned int max_width;        /**< Largest image width */
	unsigned int max_height;       /**< Largest image height */
} DatasetInfo;

/**
 * @struct MemoryInfo
 * @brief Memory usage of the process.
 *
 * Major page faults are faults that required I/O, e.g. pages brought back
 * from swap; a non-zero count means the timings were affected by memory
 * pressure. Page faults are counted over the timed trials only.
 */
typedef struct {
	double cpu_peak_rss_gb;  /**< Peak resident set size of the process in GB */
	long major_page_faults;  /**< Page faults requiring I/O (timed trials) */
	long minor_page_faults;  /**< Page faults served from memory (timed trials) */
} MemoryInfo;

/**
 * @struct Benchmark
 * @brief Master benchmark structure holding all results and metadata.
 *
 * Raw samples are accumulated trial by trial and summarized when the
 * results are written.
 */
typedef struct {
	SystemInfo sys_info;              /**< System information */
	BenchmarkInfo benchmark_info;     /**< Benchmark parameters */
	DatasetInfo dataset_info;         /**< Image set information */
	DenoiseInfo denoise_info;         /**< Denoising configuration and noise */
	EdgesConfig edges_info;           /**< Edge detection configuration */
	StageResult results[STAGE_COUNT]; /**< Per-stage results */
	Statistics pipeline_time;         /**< Whole-pipeline time over trials */
	Statistics gpu_time[GPU_METRICS]; /**< GPU times per step over trials (see GPU_* below) */
	MemoryInfo memory;                /**< Memory usage */

	/* raw samples, internal */
	unsigned int trials_done;         /**< Timed trials recorded so far */
	size_t images;                    /**< Images per trial (sample capacity) */
	double *wall[STAGE_COUNT];        /**< Stage wall time per trial */
	double *samples[STAGE_COUNT];     /**< Per-image times, all trials */
	size_t nsamples[STAGE_COUNT];     /**< Number of per-image samples */
	double *pipeline;                 /**< Whole-pipeline time per trial */
	double *gpu[GPU_METRICS];         /**< GPU times per step and launches, per trial */
	double trial_start;               /**< Start time of the current trial */
	long majflt_start;                /**< Major faults at trial start */
	long minflt_start;                /**< Minor faults at trial start */
} Benchmark;

/* ------------------------------------------------------------------------- */
/*                          Public API Functions                             */
/* ------------------------------------------------------------------------- */

/**
 * @brief Returns current monotonic time in seconds.
 *
 * Uses CLOCK_MONOTONIC for reliable timing measurements that are not
 * affected by system clock adjustments.
 *
 * @return Current time in seconds (floating point).
 */
double now_sec(void);

/**
 * @brief Initializes a benchmark structure.
 *
 * Allocates a new Benchmark and its sample buffers, records the run
 * parameters and a timestamp, and captures system information.
 *
 * @param[in] input   Input path.
 * @param[in] output  Output path, or NULL if nothing is written.
 * @param[in] threads Number of OpenMP threads.
 * @param[in] trials  Number of timed trials (> 0).
 * @param[in] wtrials Number of warmup trials.
 * @param[in] batch   Images per batch.
 * @param[in] batches Number of batches per trial.
 * @param[in] images  Number of images in the set.
 * @param[in] denoise Denoising stage configuration.
 * @param[in] edges   Edge detection stage configuration.
 *
 * @return Pointer to a newly allocated Benchmark structure, or NULL on failure.
 */
Benchmark* benchmark_init(const char *input, const char *output,
                          const unsigned int threads, const unsigned int trials,
                          const unsigned int wtrials, const size_t batch,
                          const size_t batches, const size_t images,
                          const DenoiseConfig *denoise, const EdgesConfig *edges);

/**
 * @brief Records the GPU used for the run.
 *
 * @param[in,out] b    Benchmark structure.
 * @param[in]     info GPU information.
 */
void benchmark_set_gpu(Benchmark *b, const GpuInfo *info);

/**
 * @brief Frees a Benchmark structure. Safe to call with NULL.
 *
 * @param[in,out] b Pointer to the Benchmark structure to free.
 */
void benchmark_free(Benchmark *b);

/**
 * @brief Marks the start of a timed trial.
 *
 * Records the start time and the page fault counters.
 *
 * @param[in,out] b Benchmark structure.
 */
void benchmark_trial_start(Benchmark *b);

/**
 * @brief Records a finished timed trial.
 *
 * Stores the stage wall times and per-image times from the image set, the
 * whole-pipeline time since benchmark_trial_start(), and the page faults.
 * The first call also fills the dataset information.
 *
 * @param[in,out] b   Benchmark structure.
 * @param[in]     set Image set after the trial.
 *
 * @return 0 on success, 1 on error.
 */
int benchmark_trial_end(Benchmark *b, const ImageSet *set);

/**
 * @brief Writes benchmark results as JSON.
 *
 * Summarizes the recorded trials, captures the peak memory usage, and
 * writes system information, benchmark parameters, dataset information,
 * stage results, pipeline time and memory usage.
 *
 * @param[in,out] b    Benchmark structure with recorded trials.
 * @param[in]     path Output file path, or "-" for stdout.
 *
 * @return 0 on success, 1 on error (already reported).
 */
int benchmark_write(Benchmark *b, const char *path);

#endif /* BENCHMARK_H */
