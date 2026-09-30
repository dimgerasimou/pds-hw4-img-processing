/**
 * @file args.c
 * @brief Command-line parsing.
 */

#define _GNU_SOURCE

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "args.h"
#include "error.h"
#include "image.h"

#define MAX_PATCH_RADIUS  10
#define MAX_SEARCH_RADIUS 50

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
		"Denoising (Non-Local Means):\n"
		"  -d                 Enable denoising\n"
		"  -g                 Run the filters on the GPU (CUDA); the CPU reads,\n"
		"                     decodes, encodes and writes other batches meanwhile\n"
		"  -P <radius>        Patch radius, patches are (2r+1)^2   (default: 2)\n"
		"  -S <radius>        Search radius, window is (2r+1)^2    (default: 10)\n"
		"  -H <k>             Strength, h = k * sigma (default: automatic, 0.7 for white\n"
		"                     noise up to 1.6 for correlated noise; 0.7 times that with -e)\n"
		"  -N <sigma>         Noise standard deviation (default: estimated per image)\n"
		"\n"
		"Edge detection (Canny), after denoising if both are enabled:\n"
		"  -e                 Enable; the output is the edge map (255 = edge)\n"
		"  -G <sigma>         Gaussian blur standard deviation, 0 = none (default: 1.4)\n"
		"  -l <low>           Low threshold, gradient magnitude    (default: 20)\n"
		"  -u <high>          High threshold, gradient magnitude   (default: 50)\n"
		"                       pixels above high are edges, pixels above low are\n"
		"                       edges if connected to one\n\n"
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
		"  %s -p -b results.json -f png -o out/ data/\n"
		"  %s -d -P 3 -S 7 -o clean.pgm noisy.png\n"
		"  %s -d -g -p -o clean/ xrays/\n"
		"  %s -d -e -g -o edges/ xrays/\n",
		program_name, program_name, program_name, program_name, program_name, program_name
	);

	free(program_name);
}

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

static int
parse_udouble(const char *s, double *out)
{
	char *end = NULL;
	double v;

	if (!s || !*s || *s == '-' || *s == '+')
		return 0;

	errno = 0;
	v = strtod(s, &end);

	if (errno != 0 || end == s || *end != '\0' || !isfinite(v) || v < 0.0)
		return 0;

	*out = v;
	return 1;
}

static int
bad_num(char opt)
{
	uerrf("invalid or missing argument for -%c", opt);
	usage();
	return 1;
}

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

int
parse_args(int argc, char *argv[], Args *args)
{
	int opt, repeat = 0, nlm_opt = 0, canny_opt = 0;

	if (!args) {
		DERRF("args is NULL");
		return 1;
	}

	opterr = 0;

	while ((opt = getopt(argc, argv, "o:f:t:B:b:n:w:pdgP:S:H:N:eG:l:u:h")) != -1) {
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

		case 'd':
			args->denoise = 1;
			break;

		case 'g':
			args->gpu = 1;
			break;

		case 'e':
			args->edges = 1;
			break;

		case 'G': {
			double v;
			if (!parse_udouble(optarg, &v) || v > 10.0)
				return bad_num('G');
			args->canny.sigma = v;
			canny_opt = 1;
			break;
		}

		case 'l': {
			double v;
			if (!parse_udouble(optarg, &v))
				return bad_num('l');
			args->canny.low = v;
			canny_opt = 1;
			break;
		}

		case 'u': {
			double v;
			if (!parse_udouble(optarg, &v))
				return bad_num('u');
			args->canny.high = v;
			canny_opt = 1;
			break;
		}

		case 'P': {
			unsigned int v;
			if (!parse_uint(optarg, &v))
				return bad_num('P');
			if (v > MAX_PATCH_RADIUS) {
				uerrf("patch radius must be at most %d", MAX_PATCH_RADIUS);
				usage();
				return 1;
			}
			args->nlm.patch = v;
			nlm_opt = 1;
			break;
		}

		case 'S': {
			unsigned int v;
			if (!parse_uint(optarg, &v))
				return bad_num('S');
			if (v == 0 || v > MAX_SEARCH_RADIUS) {
				uerrf("search radius must be between 1 and %d", MAX_SEARCH_RADIUS);
				usage();
				return 1;
			}
			args->nlm.search = v;
			nlm_opt = 1;
			break;
		}

		case 'H': {
			double v;
			if (!parse_udouble(optarg, &v) || v == 0.0)
				return bad_num('H');
			args->nlm.h_factor = v;
			nlm_opt = 1;
			break;
		}

		case 'N': {
			double v;
			if (!parse_udouble(optarg, &v))
				return bad_num('N');
			args->nlm.sigma = v;
			nlm_opt = 1;
			break;
		}

		case 'h':
			usage();
			return -1;

		case '?':
		default:
			if (optopt == 'o' || optopt == 'f' || optopt == 't' || optopt == 'B'
			    || optopt == 'b' || optopt == 'n' || optopt == 'w' || optopt == 'P'
			    || optopt == 'S' || optopt == 'H' || optopt == 'N' || optopt == 'G'
			    || optopt == 'l' || optopt == 'u')
				return bad_opt(optopt, 1);
			return bad_opt(optopt ? optopt : '?', 0);
		}
	}

	if (nlm_opt && !args->denoise) {
		uerrf("-P, -S, -H and -N require -d");
		usage();
		return 1;
	}

	if (canny_opt && !args->edges) {
		uerrf("-G, -l and -u require -e");
		usage();
		return 1;
	}

	if (args->gpu && !args->denoise && !args->edges) {
		uerrf("-g requires -d or -e");
		usage();
		return 1;
	}

	if (args->edges && args->canny.low > args->canny.high) {
		uerrf("low threshold (-l) must not exceed high threshold (-u)");
		usage();
		return 1;
	}

	if (args->denoise && args->edges)
		args->nlm.strength_scale = NLM_EDGE_SCALE;

	if (repeat && !args->bench_path) {
		uerrf("-n and -w require -b");
		usage();
		return 1;
	}

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
