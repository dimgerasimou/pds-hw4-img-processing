/**
 * @file error.h
 * @brief Error reporting to stderr, safe to use from multiple threads.
 *
 * uerrf()/uerrnof() are for user-facing errors ("prog: message"), the
 * DERRF()/DERRNOF() macros for internal errors, with timestamp and source
 * location. Each message is printed under a stderr lock, so concurrent
 * messages never interleave.
 */

#ifndef ERROR_H
#define ERROR_H

#include <stdarg.h>
#include <errno.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Sets the program name used in messages, from argv[0]. */
void err_init(const char *name);

/** @brief Returns a copy of the program name, which the caller frees. */
char* get_progname(void);

void uerrf(const char *fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
	__attribute__((format(printf, 1, 2)))
#endif
;

/** @brief Like uerrf(), followed by strerror(err) unless err is 0. */
void uerrnof(int err, const char *fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
	__attribute__((format(printf, 2, 3)))
#endif
;

void vderrf_at(const char *file, int line, const char *func,
               int err, const char *fmt, va_list ap);

void derrf_at(const char *file, int line, const char *func,
              int err, const char *fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
	__attribute__((format(printf, 5, 6)))
#endif
;

#define DERRF(fmt, ...)    derrf_at(__FILE__, __LINE__, __func__, 0, (fmt), ##__VA_ARGS__)
#define DERRNOF(fmt, ...)  derrf_at(__FILE__, __LINE__, __func__, errno, (fmt), ##__VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif /* ERROR_H */
