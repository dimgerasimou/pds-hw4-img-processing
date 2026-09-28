/**
 * @file json.h
 * @brief Printers for the benchmark JSON. Each prints one indented member,
 *        without a trailing comma.
 */

#ifndef JSON_H
#define JSON_H

#include <stdio.h>

#include "benchmark.h"

void print_sys_info(FILE *f, const SystemInfo *info, const unsigned int indent_level);
void print_benchmark_info(FILE *f, const BenchmarkInfo *info, const unsigned int indent_level);
void print_dataset_info(FILE *f, const DatasetInfo *info, const unsigned int indent_level);

/* The next two print null when the stage is disabled. */
void print_denoise_info(FILE *f, const DenoiseInfo *info, const unsigned int indent_level);
void print_edges_info(FILE *f, const EdgesConfig *cfg, int gpu, const unsigned int indent_level);

void print_statistics(FILE *f, const char *name, const Statistics *s,
                      const unsigned int indent_level);
void print_stage_result(FILE *f, const char *name, const StageResult *r,
                        const unsigned int indent_level);
void print_memory_info(FILE *f, const MemoryInfo *info, const unsigned int indent_level);

#endif /* JSON_H */
