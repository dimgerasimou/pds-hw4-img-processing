/**
 * @file json.c
 * @brief JSON printers for the benchmark output.
 */

#include <stdio.h>

#include "json.h"

/* Non-ASCII bytes pass through unchanged (UTF-8 stays valid). */
static void
json_print_escaped(FILE *f, const char *s)
{
	const unsigned char *p = (const unsigned char *)(s ? s : "");

	fputc('"', f);
	for (; *p; p++) {
		switch (*p) {
		case '\"': fputs("\\\"", f); break;
		case '\\': fputs("\\\\", f); break;
		case '\b': fputs("\\b",  f); break;
		case '\f': fputs("\\f",  f); break;
		case '\n': fputs("\\n",  f); break;
		case '\r': fputs("\\r",  f); break;
		case '\t': fputs("\\t",  f); break;
		default:
			if (*p < 0x20)
				fprintf(f, "\\u%04x", (unsigned)*p);
			else
				fputc(*p, f);
		}
	}
	fputc('"', f);
}

void
print_sys_info(FILE *f, const SystemInfo *info, const unsigned int indent_level)
{
	fprintf(f, "%*s\"sys_info\": {\n", indent_level, "");
	fprintf(f, "%*s\"cpu_info\": ", indent_level + 2, "");
	json_print_escaped(f, info->cpu_info);
	fprintf(f, ",\n");
	fprintf(f, "%*s\"cpu_cores\": %u,\n", indent_level + 2, "", info->cpu_cores);
	fprintf(f, "%*s\"ram_gb\": %.2f,\n", indent_level + 2, "", info->ram_gb);
	fprintf(f, "%*s\"swap_gb\": %.2f,\n", indent_level + 2, "", info->swap_gb);

	fprintf(f, "%*s\"gpu\": ", indent_level + 2, "");
	if (!info->has_gpu) {
		fputs("null\n", f);
	} else {
		fprintf(f, "{\n");
		fprintf(f, "%*s\"name\": ", indent_level + 4, "");
		json_print_escaped(f, info->gpu.name);
		fprintf(f, ",\n");
		fprintf(f, "%*s\"compute_capability\": \"%d.%d\",\n", indent_level + 4, "",
		        info->gpu.cc_major, info->gpu.cc_minor);
		fprintf(f, "%*s\"multiprocessors\": %d,\n", indent_level + 4, "", info->gpu.sm_count);
		fprintf(f, "%*s\"memory_gb\": %.2f,\n", indent_level + 4, "", info->gpu.mem_gb);
		fprintf(f, "%*s\"driver_version\": %d,\n", indent_level + 4, "", info->gpu.driver_version);
		fprintf(f, "%*s\"runtime_version\": %d\n", indent_level + 4, "", info->gpu.runtime_version);
		fprintf(f, "%*s}\n", indent_level + 2, "");
	}
	fprintf(f, "%*s}", indent_level, "");
}

void
print_benchmark_info(FILE *f, const BenchmarkInfo *info, const unsigned int indent_level)
{
	fprintf(f, "%*s\"benchmark_info\": {\n", indent_level, "");
	fprintf(f, "%*s\"timestamp\": ", indent_level + 2, "");
	json_print_escaped(f, info->timestamp);
	fprintf(f, ",\n");
	fprintf(f, "%*s\"threads\": %u,\n", indent_level + 2, "", info->threads);
	fprintf(f, "%*s\"trials\": %u,\n", indent_level + 2, "", info->trials);
	fprintf(f, "%*s\"wtrials\": %u,\n", indent_level + 2, "", info->wtrials);
	fprintf(f, "%*s\"batch_size\": %zu,\n", indent_level + 2, "", info->batch_size);
	fprintf(f, "%*s\"batches\": %zu\n", indent_level + 2, "", info->batches);
	fprintf(f, "%*s}", indent_level, "");
}

void
print_dataset_info(FILE *f, const DatasetInfo *info, const unsigned int indent_level)
{
	fprintf(f, "%*s\"dataset_info\": {\n", indent_level, "");

	fprintf(f, "%*s\"input_path\": ", indent_level + 2, "");
	json_print_escaped(f, info->input_path);
	fprintf(f, ",\n");

	fprintf(f, "%*s\"output_path\": ", indent_level + 2, "");
	if (info->output_path[0])
		json_print_escaped(f, info->output_path);
	else
		fputs("null", f);
	fprintf(f, ",\n");

	fprintf(f, "%*s\"output_format\": ", indent_level + 2, "");
	if (info->output_format != IMG_FMT_UNKNOWN)
		json_print_escaped(f, image_format_name(info->output_format));
	else
		fputs("null", f);
	fprintf(f, ",\n");

	fprintf(f, "%*s\"images\": %zu,\n", indent_level + 2, "", info->images);

	{
		int first = 1;

		fprintf(f, "%*s\"input_formats\": {", indent_level + 2, "");
		for (int fmt = 1; fmt < IMG_FMT_COUNT; fmt++) {
			if (!info->formats[fmt])
				continue;
			fprintf(f, "%s ", first ? "" : ",");
			json_print_escaped(f, image_format_name(fmt));
			fprintf(f, ": %zu", info->formats[fmt]);
			first = 0;
		}
		fprintf(f, "%s},\n", first ? "" : " ");
	}

	fprintf(f, "%*s\"pixels\": %llu,\n", indent_level + 2, "", info->pixels);
	fprintf(f, "%*s\"min_width\": %u,\n", indent_level + 2, "", info->min_width);
	fprintf(f, "%*s\"min_height\": %u,\n", indent_level + 2, "", info->min_height);
	fprintf(f, "%*s\"max_width\": %u,\n", indent_level + 2, "", info->max_width);
	fprintf(f, "%*s\"max_height\": %u\n", indent_level + 2, "", info->max_height);
	fprintf(f, "%*s}", indent_level, "");
}

void
print_denoise_info(FILE *f, const DenoiseInfo *info, const unsigned int indent_level)
{
	const DenoiseConfig *c = &info->config;

	fprintf(f, "%*s\"denoise\": ", indent_level, "");

	if (!c->enabled) {
		fputs("null", f);
		return;
	}

	fprintf(f, "{\n");
	fprintf(f, "%*s\"algorithm\": \"nlm\",\n", indent_level + 2, "");
	fprintf(f, "%*s\"device\": \"%s\",\n", indent_level + 2, "", info->gpu ? "cuda" : "cpu");
	fprintf(f, "%*s\"patch_radius\": %u,\n", indent_level + 2, "", c->params.patch);
	fprintf(f, "%*s\"search_radius\": %u,\n", indent_level + 2, "", c->params.search);
	if (c->params.h_factor > 0.0)
		fprintf(f, "%*s\"h_factor\": %.4f,\n", indent_level + 2, "", c->params.h_factor);
	else
		fprintf(f, "%*s\"h_factor\": \"auto\",\n", indent_level + 2, "");
	fprintf(f, "%*s\"sigma\": ", indent_level + 2, "");
	if (c->params.sigma >= 0.0)
		fprintf(f, "%.4f,\n", c->params.sigma);
	else if (c->params.sigma_scale != 1.0)
		fprintf(f, "\"estimated x %.4f\",\n", c->params.sigma_scale);
	else
		fputs("\"estimated\",\n", f);
	fprintf(f, "%*s\"sigma_used\": { \"mean\": %.4f, \"min\": %.4f, \"max\": %.4f },\n",
	        indent_level + 2, "", info->sigma_mean, info->sigma_min, info->sigma_max);
	fprintf(f, "%*s\"strength_used\": { \"mean\": %.4f, \"min\": %.4f, \"max\": %.4f }\n",
	        indent_level + 2, "", info->strength_mean, info->strength_min, info->strength_max);
	fprintf(f, "%*s}", indent_level, "");
}

void
print_edges_info(FILE *f, const EdgesConfig *cfg, int gpu, const unsigned int indent_level)
{
	fprintf(f, "%*s\"edges\": ", indent_level, "");

	if (!cfg->enabled) {
		fputs("null", f);
		return;
	}

	fprintf(f, "{\n");
	fprintf(f, "%*s\"algorithm\": \"canny\",\n", indent_level + 2, "");
	fprintf(f, "%*s\"device\": \"%s\",\n", indent_level + 2, "", gpu ? "cuda" : "cpu");
	fprintf(f, "%*s\"sigma\": %.4f,\n", indent_level + 2, "", cfg->params.sigma);
	fprintf(f, "%*s\"gaussian_radius\": %d,\n", indent_level + 2, "", cfg->setup.radius);
	fprintf(f, "%*s\"low\": %.4f,\n", indent_level + 2, "", cfg->params.low);
	fprintf(f, "%*s\"high\": %.4f\n", indent_level + 2, "", cfg->params.high);
	fprintf(f, "%*s}", indent_level, "");
}

void
print_statistics(FILE *f, const char *name, const Statistics *s,
                 const unsigned int indent_level)
{
	fprintf(f, "%*s", indent_level, "");
	json_print_escaped(f, name);
	fprintf(f, ": {\n");
	fprintf(f, "%*s\"mean_time_s\": %.9f,\n",   indent_level + 2, "", s->mean_time_s);
	fprintf(f, "%*s\"std_dev_s\": %.9f,\n",     indent_level + 2, "", s->std_dev_s);
	fprintf(f, "%*s\"median_time_s\": %.9f,\n", indent_level + 2, "", s->median_time_s);
	fprintf(f, "%*s\"min_time_s\": %.9f,\n",    indent_level + 2, "", s->min_time_s);
	fprintf(f, "%*s\"max_time_s\": %.9f,\n",    indent_level + 2, "", s->max_time_s);
	fprintf(f, "%*s\"total_time_s\": %.9f\n",   indent_level + 2, "", s->total_time_s);
	fprintf(f, "%*s}", indent_level, "");
}

void
print_stage_result(FILE *f, const char *name, const StageResult *r,
                   const unsigned int indent_level)
{
	fprintf(f, "%*s", indent_level, "");
	json_print_escaped(f, name);

	if (!r->performed) {
		fputs(": null", f);
		return;
	}

	fprintf(f, ": {\n");
	fprintf(f, "%*s\"images\": %zu,\n", indent_level + 2, "", r->images);
	fprintf(f, "%*s\"failed\": %zu,\n", indent_level + 2, "", r->failed);
	fprintf(f, "%*s\"data_mib\": %.4f,\n", indent_level + 2, "", r->data_mib);
	fprintf(f, "%*s\"throughput_mib_s\": %.4f,\n", indent_level + 2, "", r->throughput_mib_s);
	fprintf(f, "%*s\"images_per_sec\": %.4f,\n", indent_level + 2, "", r->images_per_sec);
	print_statistics(f, "wall_time", &r->wall_time, indent_level + 2);
	fprintf(f, ",\n");
	print_statistics(f, "per_image", &r->per_image, indent_level + 2);
	fprintf(f, "\n%*s}", indent_level, "");
}

void
print_memory_info(FILE *f, const MemoryInfo *info, const unsigned int indent_level)
{
	fprintf(f, "%*s\"memory\": {\n", indent_level, "");
	fprintf(f, "%*s\"cpu_peak_rss_gb\": %.4f,\n", indent_level + 2, "", info->cpu_peak_rss_gb);
	fprintf(f, "%*s\"major_page_faults\": %ld,\n", indent_level + 2, "", info->major_page_faults);
	fprintf(f, "%*s\"minor_page_faults\": %ld\n", indent_level + 2, "", info->minor_page_faults);
	fprintf(f, "%*s}", indent_level, "");
}
