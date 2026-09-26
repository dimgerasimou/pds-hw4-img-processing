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

#include "io.h"

/* ------------------------------------------------------------------------- */
/*                          Benchmark Data Structures                        */
/* ------------------------------------------------------------------------- */

/**
 * @struct Statistics
 * @brief Statistical summary of per-image timing results.
 */
typedef struct {
	double mean_time_s;   /**< Mean time per image in seconds */
	double std_dev_s;     /**< Standard deviation in seconds */
	double median_time_s; /**< Median time per image in seconds */
	double min_time_s;    /**< Minimum time per image in seconds */
	double max_time_s;    /**< Maximum time per image in seconds */
	double total_time_s;  /**< Sum of per-image times in seconds (CPU-side work) */
} Statistics;

/**
 * @struct StageResult
 * @brief Timing result of one stage over the whole image set.
 *
 * data_mib is the data volume of the stage as defined by io_run(): bytes
 * read, pixels produced, pixels consumed, or bytes written.
 *
 * wall_time_s is the elapsed time of the whole (parallel) stage, while
 * per_image.total_time_s is the sum of the individual image times; their
 * ratio is the effective parallelism achieved.
 */
typedef struct {
	int performed;          /**< 1 if the stage ran, else 0 */
	size_t images;          /**< Images processed successfully */
	size_t failed;          /**< Images that failed */
	double data_mib;        /**< Pixel data processed in MiB */
	double wall_time_s;     /**< Elapsed time of the whole stage in seconds */
	double throughput_mib_s;/**< data_mib / wall_time_s */
	double images_per_sec;  /**< images / wall_time_s */
	Statistics per_image;   /**< Per-image timing statistics */
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
} SystemInfo;

/**
 * @struct BenchmarkInfo
 * @brief Benchmark execution parameters.
 */
typedef struct {
	char timestamp[32];   /**< ISO 8601 timestamp of the run */
	unsigned int threads; /**< Number of OpenMP threads used */
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
 * @struct Benchmark
 * @brief Master benchmark structure holding all results and metadata.
 */
typedef struct {
	SystemInfo sys_info;          /**< System information */
	BenchmarkInfo benchmark_info; /**< Benchmark parameters */
	DatasetInfo dataset_info;     /**< Image set information */
	StageResult results[STAGE_COUNT]; /**< Per-stage results */
	double cpu_peak_rss_gb;       /**< Peak resident set size of the process in GB */
} Benchmark;

/* ------------------------------------------------------------------------- */
/*                          Public API Functions                             */
/* ------------------------------------------------------------------------- */

/**
 * @brief Returns current monotonic time in seconds.
 *
 * @return Current time in seconds (floating point).
 */
double now_sec(void);

/**
 * @brief Initializes a benchmark structure.
 *
 * Allocates a new Benchmark, records the run parameters and a timestamp,
 * and captures system information.
 *
 * @param[in] input   Input path.
 * @param[in] output  Output path, or NULL if nothing is written.
 * @param[in] threads Number of OpenMP threads.
 *
 * @return Pointer to a newly allocated Benchmark structure, or NULL on failure.
 */
Benchmark* benchmark_init(const char *input, const char *output,
                          const unsigned int threads);

/**
 * @brief Frees a Benchmark structure. Safe to call with NULL.
 *
 * @param[in,out] b Pointer to the Benchmark structure to free.
 */
void benchmark_free(Benchmark *b);

/**
 * @brief Records the results of all stages that ran.
 *
 * Computes per-image statistics from the timings stored in the image set
 * and fills the dataset information. Call once, after the last stage.
 *
 * @param[in,out] b   Benchmark structure.
 * @param[in]     set Image set after the stages ran.
 *
 * @return 0 on success, 1 on error.
 */
int benchmark_collect(Benchmark *b, const ImageSet *set);

/**
 * @brief Writes benchmark results as JSON.
 *
 * Captures the peak memory usage and writes system information, benchmark
 * parameters, dataset information and stage results.
 *
 * @param[in,out] b    Benchmark structure with populated data.
 * @param[in]     path Output file path, or "-" for stdout.
 *
 * @return 0 on success, 1 on error (already reported).
 */
int benchmark_write(Benchmark *b, const char *path);

#endif /* BENCHMARK_H */
