/**
 * @file json.c
 * @brief Minimal JSON printer for benchmark output (valid JSON).
 *
 * Provides functions to print benchmark structures in properly formatted
 * and escaped JSON to an arbitrary stream.
 */

#include <stdio.h>

#include "json.h"

/* ------------------------------------------------------------------------- */
/*                            Static Helper Functions                        */
/* ------------------------------------------------------------------------- */

/**
 * @brief Prints a JSON-escaped string with surrounding quotes.
 *
 * Escapes special characters according to JSON spec: quotes, backslashes,
 * control characters, etc. Non-ASCII bytes are passed through.
 *
 * @param[in] f Output stream.
 * @param[in] s String to print (NULL prints empty string).
 */
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

/* ------------------------------------------------------------------------- */
/*                            Public API Functions                           */
/* ------------------------------------------------------------------------- */

/**
 * @brief Print system information as formatted JSON.
 *
 * Outputs a JSON object containing the CPU model, logical core count,
 * total RAM, and swap space in gigabytes.
 *
 * @param[in] f            Output stream.
 * @param[in] info         Pointer to SystemInfo structure to print.
 * @param[in] indent_level Number of spaces to indent the output.
 */
void
print_sys_info(FILE *f, const SystemInfo *info, const unsigned int indent_level)
{
	fprintf(f, "%*s\"sys_info\": {\n", indent_level, "");
	fprintf(f, "%*s\"cpu_info\": ", indent_level + 2, "");
	json_print_escaped(f, info->cpu_info);
	fprintf(f, ",\n");
	fprintf(f, "%*s\"cpu_cores\": %u,\n", indent_level + 2, "", info->cpu_cores);
	fprintf(f, "%*s\"ram_gb\": %.2f,\n", indent_level + 2, "", info->ram_gb);
	fprintf(f, "%*s\"swap_gb\": %.2f\n", indent_level + 2, "", info->swap_gb);
	fprintf(f, "%*s}", indent_level, "");
}

/**
 * @brief Print benchmark parameters as formatted JSON.
 *
 * Outputs a JSON object containing the timestamp, the thread count, the
 * number of timed and warmup trials, and the batching parameters.
 *
 * @param[in] f            Output stream.
 * @param[in] info         Pointer to BenchmarkInfo structure to print.
 * @param[in] indent_level Number of spaces to indent the output.
 */
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

/**
 * @brief Print dataset information as formatted JSON.
 *
 * Outputs a JSON object containing the input and output paths, the output
 * format, the number of images and their input formats, total pixels and
 * the range of image dimensions.
 *
 * @param[in] f            Output stream.
 * @param[in] info         Pointer to DatasetInfo structure to print.
 * @param[in] indent_level Number of spaces to indent the output.
 */
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

	/* decoded images per input format, only formats that occurred */
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

/**
 * @brief Print a timing summary as formatted JSON.
 *
 * Outputs `"<name>": { mean, std_dev, median, min, max, total }`.
 *
 * @param[in] f            Output stream.
 * @param[in] name         JSON key.
 * @param[in] s            Pointer to Statistics structure to print.
 * @param[in] indent_level Number of spaces to indent the output.
 */
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

/**
 * @brief Print the result of one stage as formatted JSON.
 *
 * Outputs `"<name>": { ... }` with counts, data volume, throughput, and
 * the wall time and per-image time summaries, or `"<name>": null` if the
 * stage did not run.
 *
 * @param[in] f            Output stream.
 * @param[in] name         JSON key of the stage.
 * @param[in] r            Pointer to StageResult structure to print.
 * @param[in] indent_level Number of spaces to indent the output.
 */
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

/**
 * @brief Print memory usage as formatted JSON.
 *
 * Outputs a JSON object containing the peak resident set size and the
 * major and minor page faults of the timed trials.
 *
 * @param[in] f            Output stream.
 * @param[in] info         Pointer to MemoryInfo structure to print.
 * @param[in] indent_level Number of spaces to indent the output.
 */
void
print_memory_info(FILE *f, const MemoryInfo *info, const unsigned int indent_level)
{
	fprintf(f, "%*s\"memory\": {\n", indent_level, "");
	fprintf(f, "%*s\"cpu_peak_rss_gb\": %.4f,\n", indent_level + 2, "", info->cpu_peak_rss_gb);
	fprintf(f, "%*s\"major_page_faults\": %ld,\n", indent_level + 2, "", info->major_page_faults);
	fprintf(f, "%*s\"minor_page_faults\": %ld\n", indent_level + 2, "", info->minor_page_faults);
	fprintf(f, "%*s}", indent_level, "");
}
