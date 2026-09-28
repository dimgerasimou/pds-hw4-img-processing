/**
 * @file error.c
 * @brief Error reporting.
 */

#define _POSIX_C_SOURCE 200809L

#include "error.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

static const char *g_prog = "imgfilter";

static void
set_progname(const char *name)
{
	if (!name || !*name)
		return;

	const char *slash = strrchr(name, '/');
	g_prog = slash ? slash + 1 : name;
}

static void
timestamp_now(char *buf, size_t n)
{
	time_t t = time(NULL);
	struct tm tm;

	if (n == 0)
		return;

	if (localtime_r(&t, &tm))
		strftime(buf, n, "%Y-%m-%d %H:%M:%S", &tm);
	else
		buf[0] = '\0';
}

void
err_init(const char *name)
{
	set_progname(name);
}

char*
get_progname(void)
{
	char *ret = strdup(g_prog);
	if (!ret) {
		DERRNOF("strdup failed");
		return NULL;
	}
	return ret;
}

void
uerrf(const char *fmt, ...)
{
	va_list ap;

	if (!fmt)
		fmt = "?";

	flockfile(stderr);
	fprintf(stderr, "%s: ", g_prog);

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);

	fputc('\n', stderr);
	funlockfile(stderr);
}

void
uerrnof(int err, const char *fmt, ...)
{
	va_list ap;

	if (!fmt)
		fmt = "?";

	flockfile(stderr);
	fprintf(stderr, "%s: ", g_prog);

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);

	if (err)
		fprintf(stderr, ": %s\n", strerror(err));
	else
		fputc('\n', stderr);
	funlockfile(stderr);
}

void
vderrf_at(const char *file, int line, const char *func,
          int err, const char *fmt, va_list ap)
{
	char tbuf[32];

	if (!file) file = "?";
	if (!func) func = "?";
	if (!fmt)  fmt  = "?";

	timestamp_now(tbuf, sizeof(tbuf));

	flockfile(stderr);
	fprintf(stderr, "[%s] %s %s:%d %s: ", tbuf, g_prog, file, line, func);
	vfprintf(stderr, fmt, ap);

	if (err)
		fprintf(stderr, " (errno=%d: %s)\n", err, strerror(err));
	else
		fputc('\n', stderr);
	funlockfile(stderr);
}

void
derrf_at(const char *file, int line, const char *func,
         int err, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vderrf_at(file, line, func, err, fmt, ap);
	va_end(ap);
}
