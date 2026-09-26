/**
 * @file args.c
 * @brief Command-line argument parsing implementation.
 *
 * Provides functions to parse program arguments.
 */

/*
 * _GNU_SOURCE rather than _POSIX_C_SOURCE: with _POSIX_C_SOURCE alone glibc
 * maps getopt() to its strict POSIX variant, which stops at the first
 * operand, so "imgfilter data/ -o out/" would fail. The GNU variant permutes
 * arguments like the coreutils do; setting POSIXLY_CORRECT in the
 * environment restores strict POSIX behaviour.
 */
#define _GNU_SOURCE

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "args.h"
#include "error.h"
#include "image.h"

/* ------------------------------------------------------------------------- */
/*                            Static Helper Functions                        */
/* ------------------------------------------------------------------------- */

/**
 * @brief Prints program usage instructions.
 *
 * Displays the valid command-line options and their expected arguments.
 */
static void
usage(void)
{
	char *program_name = get_progname();

	if (!program_name)
		return;

	printf(
		"\n"
		"Usage: %s [OPTIONS] <input>\n\n"
		"Options:\n"
		"  -o <output>        Output file or directory (default: none)\n"
		"  -f <format>        Output format: pgm, png  (default: pgm, or the\n"
		"                     extension of a single -o file)\n"
		"  -t <threads>       Number of threads        (default: all cores)\n"
		"  -B <images>        Images per batch, 0 = all at once (default: 256)\n"
		"  -b <file>          Write benchmark JSON to file, '-' for stdout\n"
		"  -n <trials>        Timed benchmark trials   (default: 1, needs -b)\n"
		"  -w <wtrials>       Warmup benchmark trials  (default: 0, needs -b)\n"
		"  -p                 Show progress bars\n"
		"  -h                 Show this help message and exit\n\n"
		"Arguments:\n"
		"  input              Image or directory of images\n"
		"                     (PGM, PNG, JPEG, BMP, TGA; converted to grayscale)\n\n"
		"Input/output:\n"
		"  file -> file       Process one image into the given file\n"
		"  file -> dir        Write into dir/, keeping the input base name\n"
		"  dir  -> dir        Process every image, keeping base names\n"
		"                     (dir is created if it does not exist)\n"
		"  no -o              Process only, nothing is written\n\n"
		"Examples:\n"
		"  %s -t 8 -o out/ data/\n"
		"  %s -p -b results.json -f png -o out/ data/\n",
		program_name, program_name, program_name
	);

	free(program_name);
}

/**
 * @brief Parse an unsigned int from a decimal string with full validation.
 *
 * Accepts only a base-10 non-negative integer (no leading sign, no trailing
 * garbage). Detects overflow and reports failure.
 *
 * @param[in]  s   Input string.
 * @param[out] out Output value.
 * @return 1 on success, 0 on failure.
 */
static int
parse_uint(const char *s, unsigned int *out)
{
	char *end = NULL;
	unsigned long v;

	if (!s || !*s || *s == '-' || *s == '+')
		return 0;

	errno = 0;
	v = strtoul(s, &end, 10);

	if (errno != 0)
		return 0;
	if (end == s || *end != '\0')
		return 0;
	if (v > UINT_MAX)
		return 0;

	*out = (unsigned int)v;
	return 1;
}

/**
 * @brief Emit a consistent error for invalid/missing numeric argument.
 */
static int
bad_num(char opt)
{
	uerrf("invalid or missing argument for -%c", opt);
	usage();
	return 1;
}

/**
 * @brief Emit a consistent error for missing/unknown option.
 */
static int
bad_opt(int opt, int is_missing_arg)
{
	if (is_missing_arg)
		uerrf("missing argument for -%c", opt);
	else
		uerrf("unknown option '-%c'", opt ? opt : '?');
	usage();
	return 1;
}

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
 *   -B <images>   Images per batch (0: all at once)
 *   -b <file>     Write benchmark results as JSON ("-" for stdout)
 *   -n <trials>   Timed benchmark trials (must be > 0, requires -b)
 *   -w <wtrials>  Warmup benchmark trials (requires -b)
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
int
parse_args(int argc, char *argv[], Args *args)
{
	int opt, repeat = 0;

	if (!args) {
		DERRF("args is NULL");
		return 1;
	}

	opterr = 0;

	while ((opt = getopt(argc, argv, "o:f:t:B:b:n:w:ph")) != -1) {
		switch (opt) {
		case 'o':
			args->output = optarg;
			break;

		case 'f': {
			int f = image_format_from_name(optarg);
			if (!image_format_writable(f)) {
				uerrf("unsupported output format '%s' (use pgm or png)", optarg);
				usage();
				return 1;
			}
			args->format = f;
			break;
		}

		case 't': {
			unsigned int v;
			if (!parse_uint(optarg, &v))
				return bad_num('t');
			if (v == 0) {
				uerrf("threads must be > 0");
				usage();
				return 1;
			}
			args->threads = v;
			break;
		}

		case 'B': {
			unsigned int v;
			if (!parse_uint(optarg, &v))
				return bad_num('B');
			args->batch = v;
			break;
		}

		case 'b':
			args->bench_path = optarg;
			break;

		case 'n': {
			unsigned int v;
			if (!parse_uint(optarg, &v))
				return bad_num('n');
			if (v == 0) {
				uerrf("trials must be > 0");
				usage();
				return 1;
			}
			args->trials = v;
			repeat = 1;
			break;
		}

		case 'w': {
			unsigned int v;
			if (!parse_uint(optarg, &v))
				return bad_num('w');
			args->wtrials = v;
			repeat = 1;
			break;
		}

		case 'p':
			args->progress = 1;
			break;

		case 'h':
			usage();
			return -1;

		case '?':
		default:
			if (optopt == 'o' || optopt == 'f' || optopt == 't' || optopt == 'B'
			    || optopt == 'b' || optopt == 'n' || optopt == 'w')
				return bad_opt(optopt, 1);
			return bad_opt(optopt ? optopt : '?', 0);
		}
	}

	/* Repeating the work only makes sense when it is being measured */
	if (repeat && !args->bench_path) {
		uerrf("-n and -w require -b");
		usage();
		return 1;
	}

	/* Expect exactly one positional argument: the input path */
	if (optind >= argc) {
		uerrf("no input specified");
		usage();
		return 1;
	}

	if (optind + 1 < argc) {
		uerrf("too many arguments");
		usage();
		return 1;
	}

	args->input = argv[optind];

	if (access(args->input, R_OK) != 0) {
		uerrnof(errno, "cannot access \"%s\"", args->input);
		return 1;
	}

	return 0;
}
