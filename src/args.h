/**
 * @file args.h
 * @brief Command-line parsing (POSIX getopt).
 */

#ifndef ARGS_H
#define ARGS_H

#include "canny.h"
#include "nlm.h"

typedef struct {
	char *input;
	char *output;
	char *bench_path;
	int format;
	unsigned int threads;
	unsigned int batch;
	unsigned int trials;
	unsigned int wtrials;
	int progress;
	int denoise;
	int gpu;
	int edges;
	CannyParams canny;
	NlmParams nlm;
} Args;

/**
 * @brief Parses the command line into @p args, which holds the defaults.
 *
 * Reports errors and prints the usage on invalid input.
 *
 * @return 0 on success, -1 if help was requested, 1 on error.
 */
int parse_args(int argc, char *argv[], Args *args);

#endif /* ARGS_H */
