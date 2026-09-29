/**
 * @file benchmark.c
 * @brief Benchmark statistics and JSON output.
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

#define JSON_INDENT 2

static int
cmp_double(const void *a, const void *b)
{
	double da = *(const double *)a;
	double db = *(const double *)b;
	return (da > db) - (da < db);
}

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

static void
gettimestamp(Benchmark *b)
{
	time_t t = time(NULL);
	struct tm tm;
	localtime_r(&t, &tm);
	strftime(b->benchmark_info.timestamp, sizeof(b->benchmark_info.timestamp),
	         "%Y-%m-%dT%H:%M:%S", &tm);
}

static void
getpeakrss(Benchmark *b)
{
	struct rusage ru;

	/* ru_maxrss is reported in kiB */
	if (getrusage(RUSAGE_SELF, &ru) == 0)
		b->memory.cpu_peak_rss_gb = (double)ru.ru_maxrss / 1024.0 / 1024.0;
	else
		b->memory.cpu_peak_rss_gb = 0.0;
}

static void
getfaults(long *major, long *minor)
{
	struct rusage ru;

	if (getrusage(RUSAGE_SELF, &ru) == 0) {
		*major = ru.ru_majflt;
		*minor = ru.ru_minflt;
	} else {
		*major = *minor = 0;
	}
}

static void
getdatasetinfo(Benchmark *b, const ImageSet *set)
{
	DatasetInfo *d = &b->dataset_info;
	int first = 1;

	d->images = set->count;
	d->output_format = set->out_format;
	d->pixels = 0;
	memset(d->formats, 0, sizeof(d->formats));

	for (size_t i = 0; i < set->count; i++) {
		const ImageItem *it = &set->items[i];

		if (it->err[STAGE_DECODE] != IMG_OK)
			continue;

		if (it->in_format >= 0 && it->in_format < IMG_FMT_COUNT)
			d->formats[it->in_format]++;

		d->pixels += (unsigned long long)it->width * it->height;

		if (first || it->width < d->min_width)   d->min_width = it->width;
		if (first || it->height < d->min_height) d->min_height = it->height;
		if (first || it->width > d->max_width)   d->max_width = it->width;
		if (first || it->height > d->max_height) d->max_height = it->height;
		first = 0;
	}
}

static void
getdenoiseinfo(Benchmark *b, const ImageSet *set)
{
	DenoiseInfo *d = &b->denoise_info;
	double sum = 0.0, hsum = 0.0;
	size_t n = 0;

	for (size_t i = 0; i < set->count; i++) {
		const ImageItem *it = &set->items[i];

		if (it->err[STAGE_DENOISE] != IMG_OK)
			continue;

		if (n == 0 || it->sigma < d->sigma_min) d->sigma_min = it->sigma;
		if (n == 0 || it->sigma > d->sigma_max) d->sigma_max = it->sigma;
		if (n == 0 || it->strength < d->strength_min) d->strength_min = it->strength;
		if (n == 0 || it->strength > d->strength_max) d->strength_max = it->strength;
		sum += it->sigma;
		hsum += it->strength;
		n++;
	}

	d->sigma_mean = n ? sum / n : 0.0;
	d->strength_mean = n ? hsum / n : 0.0;
}

static void
summarize(Benchmark *b)
{
	unsigned int n = b->trials_done;

	for (int s = 0; s < STAGE_COUNT; s++) {
		StageResult *r = &b->results[s];
		double median;

		if (!r->performed)
			continue;

		calcstatistics(b->wall[s], n, &r->wall_time);
		calcstatistics(b->samples[s], b->nsamples[s], &r->per_image);

		median = r->wall_time.median_time_s;
		r->throughput_mib_s = median > 0.0 ? r->data_mib / median : 0.0;
		r->images_per_sec = median > 0.0 ? r->images / median : 0.0;
	}

	calcstatistics(b->pipeline, n, &b->pipeline_time);

	for (int k = 0; k < GPU_METRICS; k++)
		calcstatistics(b->gpu[k], n, &b->gpu_time[k]);
}

static void
copy_str(char *dst, size_t n, const char *src)
{
	snprintf(dst, n, "%s", src ? src : "");
}

double
now_sec(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec + t.tv_nsec / 1e9;
}

Benchmark*
benchmark_init(const char *input, const char *output, const unsigned int threads,
               const unsigned int trials, const unsigned int wtrials,
               const size_t batch, const size_t batches, const size_t images,
               const DenoiseConfig *denoise, const EdgesConfig *edges)
{
	Benchmark *b;

	if (trials == 0) {
		DERRF("trials must be > 0");
		return NULL;
	}

	b = calloc(1, sizeof(Benchmark));
	if (!b) {
		DERRNOF("calloc() failed");
		return NULL;
	}

	b->images = images;

	for (int s = 0; s < STAGE_COUNT; s++) {
		b->wall[s] = calloc(trials, sizeof(double));
		b->samples[s] = calloc((size_t)trials * (images ? images : 1), sizeof(double));
		if (!b->wall[s] || !b->samples[s])
			goto fail;
	}

	b->pipeline = calloc(trials, sizeof(double));
	if (!b->pipeline)
		goto fail;

	for (int k = 0; k < GPU_METRICS; k++) {
		b->gpu[k] = calloc(trials, sizeof(double));
		if (!b->gpu[k])
			goto fail;
	}

	copy_str(b->dataset_info.input_path, sizeof(b->dataset_info.input_path), input);
	copy_str(b->dataset_info.output_path, sizeof(b->dataset_info.output_path), output);

	b->benchmark_info.threads = threads;
	b->benchmark_info.trials = trials;
	b->benchmark_info.wtrials = wtrials;
	b->benchmark_info.batch_size = batch;
	b->benchmark_info.batches = batches;
	if (denoise)
		b->denoise_info.config = *denoise;
	if (edges)
		b->edges_info = *edges;
	gettimestamp(b);
	getcpuinfo(b);
	getmeminfo(b);

	return b;

fail:
	DERRNOF("calloc() failed");
	benchmark_free(b);
	return NULL;
}

void
benchmark_set_gpu(Benchmark *b, const GpuInfo *info)
{
	if (!b || !info)
		return;

	b->sys_info.has_gpu = 1;
	b->sys_info.gpu = *info;
	b->denoise_info.gpu = 1;
}

void
benchmark_free(Benchmark *b)
{
	if (!b)
		return;

	for (int s = 0; s < STAGE_COUNT; s++) {
		free(b->wall[s]);
		free(b->samples[s]);
	}
	free(b->pipeline);
	for (int k = 0; k < GPU_METRICS; k++)
		free(b->gpu[k]);
	free(b);
}

void
benchmark_trial_start(Benchmark *b)
{
	getfaults(&b->majflt_start, &b->minflt_start);
	b->trial_start = now_sec();
}

int
benchmark_trial_end(Benchmark *b, const ImageSet *set)
{
	double elapsed;
	unsigned int t;
	long major, minor;

	if (!b || !set) {
		DERRF("benchmark or image set not initialized");
		return 1;
	}

	elapsed = now_sec() - b->trial_start;

	if (b->trials_done >= b->benchmark_info.trials || set->count > b->images) {
		DERRF("more trials or images than allocated");
		return 1;
	}

	getfaults(&major, &minor);
	b->memory.major_page_faults += major - b->majflt_start;
	b->memory.minor_page_faults += minor - b->minflt_start;

	t = b->trials_done++;
	b->pipeline[t] = elapsed;
	b->gpu[GPU_UPLOAD][t] = set->gpu_time.upload_s;
	b->gpu[GPU_DENOISE][t] = set->gpu_time.denoise_s;
	b->gpu[GPU_EDGES][t] = set->gpu_time.edges_s;
	b->gpu[GPU_HYSTERESIS][t] = set->gpu_time.hysteresis_s;
	b->gpu[GPU_DOWNLOAD][t] = set->gpu_time.download_s;
	b->gpu[GPU_LAUNCHES][t] = set->gpu_time.launches;
	b->gpu[GPU_TILES][t] = set->gpu_time.tiles;
	b->gpu[GPU_TILES_ALL][t] = set->gpu_time.tiles_all;

	for (int s = 0; s < STAGE_COUNT; s++) {
		StageResult *r = &b->results[s];
		double bytes = 0.0;
		size_t ok = 0, failed = 0;

		if (!set->performed[s])
			continue;

		b->wall[s][t] = set->wall_time_s[s];

		for (size_t i = 0; i < set->count; i++) {
			const ImageItem *it = &set->items[i];

			if (it->err[s] == IO_NOT_DONE)
				continue;
			if (it->err[s] != IMG_OK) {
				failed++;
				continue;
			}

			b->samples[s][b->nsamples[s]++] = it->time_s[s];
			bytes += (double)it->bytes[s];
			ok++;
		}

		r->performed = 1;
		r->images = ok;
		r->failed = failed;
		r->data_mib = bytes / 1024.0 / 1024.0;
	}

	if (t == 0) {
		getdatasetinfo(b, set);
		getdenoiseinfo(b, set);
	}

	return 0;
}

int
benchmark_write(Benchmark *b, const char *path)
{
	FILE *f;
	int to_stdout;

	if (!b || !path) {
		DERRF("benchmark or path not initialized");
		return 1;
	}

	if (b->trials_done == 0) {
		DERRF("no trials recorded");
		return 1;
	}

	summarize(b);
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
	print_denoise_info(f, &b->denoise_info, JSON_INDENT);
	fprintf(f, ",\n");
	print_edges_info(f, &b->edges_info, b->denoise_info.gpu, JSON_INDENT);
	fprintf(f, ",\n");
	fprintf(f, "%*s\"results\": {\n", JSON_INDENT, "");
	for (int s = 0; s < STAGE_COUNT; s++) {
		print_stage_result(f, io_stage_name(s), &b->results[s], JSON_INDENT + 2);
		fprintf(f, s + 1 < STAGE_COUNT ? ",\n" : "\n");
	}
	fprintf(f, "%*s},\n", JSON_INDENT, "");
	print_statistics(f, "pipeline_time", &b->pipeline_time, JSON_INDENT);
	fprintf(f, ",\n");

	fprintf(f, "%*s\"gpu_time\": ", JSON_INDENT, "");
	if (!b->denoise_info.gpu) {
		fputs("null,\n", f);
	} else {
		fprintf(f, "{\n");
		print_statistics(f, "upload", &b->gpu_time[GPU_UPLOAD], JSON_INDENT + 2);
		fprintf(f, ",\n");
		print_statistics(f, "denoise_kernel", &b->gpu_time[GPU_DENOISE], JSON_INDENT + 2);
		fprintf(f, ",\n");
		print_statistics(f, "edges_kernels", &b->gpu_time[GPU_EDGES], JSON_INDENT + 2);
		fprintf(f, ",\n");
		print_statistics(f, "hysteresis", &b->gpu_time[GPU_HYSTERESIS], JSON_INDENT + 2);
		fprintf(f, ",\n");
		print_statistics(f, "download", &b->gpu_time[GPU_DOWNLOAD], JSON_INDENT + 2);
		fprintf(f, ",\n%*s\"hysteresis_launches\": %.0f,\n", JSON_INDENT + 2, "",
		        b->gpu_time[GPU_LAUNCHES].median_time_s);
		fprintf(f, "%*s\"hysteresis_tiles\": %.0f,\n", JSON_INDENT + 2, "",
		        b->gpu_time[GPU_TILES].median_time_s);
		fprintf(f, "%*s\"hysteresis_tiles_if_all_active\": %.0f\n", JSON_INDENT + 2, "",
		        b->gpu_time[GPU_TILES_ALL].median_time_s);
		fprintf(f, "%*s},\n", JSON_INDENT, "");
	}
	print_memory_info(f, &b->memory, JSON_INDENT);
	fprintf(f, "\n}\n");

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
