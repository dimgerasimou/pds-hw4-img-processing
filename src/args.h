/**
 * @file args.h
 * @brief Command-line argument parsing interface.
 *
 * Supports standard POSIX option parsing.
 */

#ifndef ARGS_H
#define ARGS_H

/* ------------------------------------------------------------------------- */
/*                              Data Structures                              */
/* ------------------------------------------------------------------------- */

/**
 * @struct Args
 * @brief Program configuration parsed from the command line.
 *
 * The caller initializes the structure with default values before calling
 * parse_args(); only options present on the command line are overwritten.
 */
typedef struct {
	char *input;          /**< Input image file or directory (required) */
	char *output;         /**< Output file or directory (-o), NULL: nothing written */
	char *bench_path;     /**< Benchmark JSON path (-b), "-" for stdout, NULL: disabled */
	int format;           /**< Output format (-f), IMG_FMT_UNKNOWN: automatic */
	unsigned int threads; /**< Number of OpenMP threads (-t) */
	int progress;         /**< Non-zero to show progress bars (-p) */
} Args;

/* ------------------------------------------------------------------------- */
/*                            Public API Functions                           */
/* ------------------------------------------------------------------------- */

/**
 * @brief Parses command-line arguments.
 *
 * Validates all arguments and ensures the input path exists and is readable.
 * Provides helpful error messages and usage information on invalid input.
 *
 * Supported options:
 *   -o <output>   Output file or directory
 *   -f <format>   Output format: pgm or png
 *   -t <threads>  Number of threads (must be > 0)
 *   -b <file>     Write benchmark results as JSON ("-" for stdout)
 *   -p            Show progress bars
 *   -h            Show usage and exit
 *
 * Required argument:
 *   <input>       Image file or directory of images
 *
 * @param[in]     argc Argument count from main().
 * @param[in]     argv Argument vector from main().
 * @param[in,out] args Configuration, pre-filled with defaults.
 *
 * @return 0 on success, -1 if help requested, 1 on error.
 */
int parse_args(int argc, char *argv[], Args *args);

#endif /* ARGS_H */
