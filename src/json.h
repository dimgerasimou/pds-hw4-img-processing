/**
 * @file json.h
 * @brief Minimal JSON printer for benchmark output.
 *
 * Provides functions to print benchmark structures as valid, properly
 * formatted and escaped JSON to a stream. Designed for easy integration
 * with analysis pipelines and result collection systems.
 */

#ifndef JSON_H
#define JSON_H

#include <stdio.h>

#include "benchmark.h"

/**
 * @brief Print system information as formatted JSON.
 *
 * Outputs a JSON object containing the CPU model, logical core count,
 * total RAM, swap space in gigabytes, and the GPU used (or null).
 *
 * @param[in] f            Output stream.
 * @param[in] info         Pointer to SystemInfo structure to print.
 * @param[in] indent_level Number of spaces to indent the output.
 */
void print_sys_info(FILE *f, const SystemInfo *info, const unsigned int indent_level);

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
void print_benchmark_info(FILE *f, const BenchmarkInfo *info, const unsigned int indent_level);

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
void print_dataset_info(FILE *f, const DatasetInfo *info, const unsigned int indent_level);

/**
 * @brief Print the denoising configuration as formatted JSON.
 *
 * Outputs `"denoise": null` if the stage is disabled, otherwise its
 * parameters and the noise standard deviations used (given, or estimated
 * per image).
 *
 * @param[in] f            Output stream.
 * @param[in] info         Pointer to DenoiseInfo structure to print.
 * @param[in] indent_level Number of spaces to indent the output.
 */
void print_denoise_info(FILE *f, const DenoiseInfo *info, const unsigned int indent_level);

/**
 * @brief Print the edge detection configuration as formatted JSON.
 *
 * Outputs `"edges": null` if the stage is disabled, otherwise the device and
 * the detector parameters (with the Gaussian radius used).
 *
 * @param[in] f            Output stream.
 * @param[in] cfg          Pointer to EdgesConfig structure to print.
 * @param[in] gpu          Non-zero if the filters ran on the GPU.
 * @param[in] indent_level Number of spaces to indent the output.
 */
void print_edges_info(FILE *f, const EdgesConfig *cfg, int gpu, const unsigned int indent_level);

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
void print_statistics(FILE *f, const char *name, const Statistics *s,
                      const unsigned int indent_level);

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
void print_stage_result(FILE *f, const char *name, const StageResult *r,
                        const unsigned int indent_level);

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
void print_memory_info(FILE *f, const MemoryInfo *info, const unsigned int indent_level);

#endif /* JSON_H */
