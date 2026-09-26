/**
 * @file benchmark.c
 * @brief Implementation of the benchmarking framework.
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <time.h>
#include <unistd.h>

#include "benchmark.h"
#include "error.h"
#include "json.h"

/* JSON output indentation level */
#define JSON_INDENT 2

/* ------------------------------------------------------------------------- */
/*                            Static Helper Functions                        */
/* ------------------------------------------------------------------------- */

/**
 * @brief Comparison function for sorting doubles.
 */
static int
cmp_double(const void *a, const void *b)
{
	double da = *(const double *)a;
	double db = *(const double *)b;
	return (da > db) - (da < db);
}

/**
 * @brief Calculates statistics over an array of times.
 *
 * Sorts @p times in place.
 *
 * @param[in,out] times Array of times in seconds.
 * @param[in]     n     Number of times.
 * @param[out]    s     Statistics to populate.
 */
static void
calcstatistics(double *times, const size_t n, Statistics *s)
{
	double sum = 0.0, sum_sq = 0.0, mean;

	memset(s, 0, sizeof(*s));

	if (n == 0)
		return;

	qsort(times, n, sizeof(double), cmp_double);

	s->min_time_s = times[0];
	s->max_time_s = times[n - 1];
	s->median_time_s = (n % 2)
		? times[n / 2]
		: (times[n / 2] + times[n / 2 - 1]) / 2.0;

	for (size_t i = 0; i < n; i++) {
		sum += times[i];
		sum_sq += times[i] * times[i];
	}

	mean = sum / n;
	s->mean_time_s = mean;
	s->total_time_s = sum;
	s->std_dev_s = (n > 1)
		? sqrt(fmax(0.0, (sum_sq - n * mean * mean) / (n - 1)))
		: 0.0;
}

/**
 * @brief Retrieves system memory information in GB.
 */
static void
getmeminfo(Benchmark *b)
{
#ifdef __linux__
	struct sysinfo info;
	if (sysinfo(&info) == 0) {
		b->sys_info.ram_gb  = info.totalram  / 1024.0 / 1024.0 / 1024.0 * info.mem_unit;
		b->sys_info.swap_gb = info.totalswap / 1024.0 / 1024.0 / 1024.0 * info.mem_unit;
	} else {
		b->sys_info.ram_gb = b->sys_info.swap_gb = 0.0;
	}
#else
	b->sys_info.ram_gb = b->sys_info.swap_gb = 0.0;
#endif
}

/**
 * @brief Retrieves CPU model information from /proc/cpuinfo.
 */
static void
getcpuinfo(Benchmark *b)
{
	long cores = sysconf(_SC_NPROCESSORS_ONLN);
	b->sys_info.cpu_cores = cores > 0 ? (unsigned int)cores : 0;

	snprintf(b->sys_info.cpu_info, sizeof(b->sys_info.cpu_info), "unknown");

#ifdef __linux__
	FILE *f = fopen("/proc/cpuinfo", "r");
	if (!f)
		return;

	char line[256];
	while (fgets(line, sizeof(line), f)) {
		if (strncmp(line, "model name", 10) == 0) {
			char *p = strchr(line, ':');
			if (p) {
				snprintf(b->sys_info.cpu_info, sizeof(b->sys_info.cpu_info), "%s", p + 2);
				b->sys_info.cpu_info[strcspn(b->sys_info.cpu_info, "\n")] = 0;
			}
			break;
		}
	}
	fclose(f);
#endif
}

/**
 * @brief Generates an ISO-8601 formatted timestamp.
 */
static void
gettimestamp(Benchmark *b)
{
	time_t t = time(NULL);
	struct tm tm;
	localtime_r(&t, &tm);
	strftime(b->benchmark_info.timestamp, sizeof(b->benchmark_info.timestamp),
	         "%Y-%m-%dT%H:%M:%S", &tm);
}

/**
 * @brief Records the peak resident set size of the process.
 */
static void
getpeakrss(Benchmark *b)
{
	struct rusage ru;

	/* ru_maxrss is reported in kib */
	if (getrusage(RUSAGE_SELF, &ru) == 0)
		b->cpu_peak_rss_gb = (double)ru.ru_maxrss / 1024.0 / 1024.0;
	else
		b->cpu_peak_rss_gb = 0.0;
}

/**
 * @brief Copies a string into a fixed buffer, always NULL-terminating.
 */
static void
copy_str(char *dst, size_t n, const char *src)
{
	snprintf(dst, n, "%s", src ? src : "");
}

/* ------------------------------------------------------------------------- */
/*                            Public API Implementation                      */
/* ------------------------------------------------------------------------- */

/**
 * @brief Returns current monotonic time in seconds.
 *
 * @return Current time in seconds (floating point).
 */
double
now_sec(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec + t.tv_nsec / 1e9;
}

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
Benchmark*
benchmark_init(const char *input, const char *output, const unsigned int threads)
{
	Benchmark *b = calloc(1, sizeof(Benchmark));
	if (!b) {
		DERRNOF("calloc() failed");
		return NULL;
	}

	copy_str(b->dataset_info.input_path, sizeof(b->dataset_info.input_path), input);
	copy_str(b->dataset_info.output_path, sizeof(b->dataset_info.output_path), output);

	b->benchmark_info.threads = threads;
	gettimestamp(b);
	getcpuinfo(b);
	getmeminfo(b);

	return b;
}

/**
 * @brief Frees a Benchmark structure. Safe to call with NULL.
 *
 * @param[in,out] b Pointer to the Benchmark structure to free.
 */
void
benchmark_free(Benchmark *b)
{
	free(b);
}

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
int
benchmark_collect(Benchmark *b, const ImageSet *set)
{
	DatasetInfo *d;
	double *times;
	int first = 1;

	if (!b || !set) {
		DERRF("benchmark or image set not initialized");
		return 1;
	}

	times = malloc((set->count ? set->count : 1) * sizeof(double));
	if (!times) {
		DERRNOF("malloc() failed");
		return 1;
	}

	for (int s = 0; s < STAGE_COUNT; s++) {
		StageResult *r = &b->results[s];
		double bytes = 0.0;
		size_t n = 0, failed = 0;

		memset(r, 0, sizeof(*r));

		if (!set->performed[s])
			continue;

		for (size_t i = 0; i < set->count; i++) {
			const ImageItem *it = &set->items[i];

			if (it->err[s] == IO_NOT_DONE)
				continue;
			if (it->err[s] != IMG_OK) {
				failed++;
				continue;
			}

			times[n++] = it->time_s[s];
			bytes += (double)it->bytes[s];
		}

		r->performed = 1;
		r->images = n;
		r->failed = failed;
		r->data_mib = bytes / 1024.0 / 1024.0;
		r->wall_time_s = set->wall_time_s[s];
		r->throughput_mib_s = r->wall_time_s > 0.0 ? r->data_mib / r->wall_time_s : 0.0;
		r->images_per_sec = r->wall_time_s > 0.0 ? n / r->wall_time_s : 0.0;
		calcstatistics(times, n, &r->per_image);
	}

	free(times);

	d = &b->dataset_info;
	d->images = set->count;
	d->output_format = set->out_format;
	d->pixels = 0;
	memset(d->formats, 0, sizeof(d->formats));

	for (size_t i = 0; i < set->count; i++) {
		const ImageItem *it = &set->items[i];
		const Image *img = &it->img;

		if (it->err[STAGE_DECODE] != IMG_OK)
			continue;

		if (it->in_format >= 0 && it->in_format < IMG_FMT_COUNT)
			d->formats[it->in_format]++;

		d->pixels += (unsigned long long)img->width * img->height;

		if (first || img->width < d->min_width)   d->min_width = img->width;
		if (first || img->height < d->min_height) d->min_height = img->height;
		if (first || img->width > d->max_width)   d->max_width = img->width;
		if (first || img->height > d->max_height) d->max_height = img->height;
		first = 0;
	}

	return 0;
}

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
int
benchmark_write(Benchmark *b, const char *path)
{
	FILE *f;
	int to_stdout;

	if (!b || !path) {
		DERRF("benchmark or path not initialized");
		return 1;
	}

	getpeakrss(b);

	to_stdout = (strcmp(path, "-") == 0);
	f = to_stdout ? stdout : fopen(path, "w");
	if (!f) {
		uerrnof(errno, "cannot open \"%s\" for writing", path);
		return 1;
	}

	fprintf(f, "{\n");
	print_sys_info(f, &b->sys_info, JSON_INDENT);
	fprintf(f, ",\n");
	print_benchmark_info(f, &b->benchmark_info, JSON_INDENT);
	fprintf(f, ",\n");
	print_dataset_info(f, &b->dataset_info, JSON_INDENT);
	fprintf(f, ",\n");
	fprintf(f, "%*s\"results\": {\n", JSON_INDENT, "");
	for (int s = 0; s < STAGE_COUNT; s++) {
		print_stage_result(f, io_stage_name(s), &b->results[s], JSON_INDENT + 2);
		fprintf(f, s + 1 < STAGE_COUNT ? ",\n" : "\n");
	}
	fprintf(f, "%*s},\n", JSON_INDENT, "");
	fprintf(f, "%*s\"cpu_peak_rss_gb\": %.4f\n", JSON_INDENT, "", b->cpu_peak_rss_gb);
	fprintf(f, "}\n");

	if (to_stdout) {
		if (fflush(f) != 0) {
			uerrnof(errno, "cannot write benchmark to stdout");
			return 1;
		}
		return 0;
	}

	{
		int bad = ferror(f);

		if (fclose(f) != 0)
			bad = 1;

		if (bad) {
			uerrnof(errno, "cannot write \"%s\"", path);
			return 1;
		}
	}

	return 0;
}
