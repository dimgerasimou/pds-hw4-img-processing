/**
 * @file progress.h
 * @brief Thread-safe terminal progress bar.
 *
 * A generic progress counter meant to be advanced from inside OpenMP
 * parallel loops (reading, writing, and later the filters themselves).
 * The bar is drawn to stderr so it never mixes with data written to stdout,
 * and it is redrawn only when the displayed percentage changes, so ticking
 * it from a hot loop costs one atomic increment per call.
 *
 * Output format:
 *   read   [=================>            ]  58%  580/1000  1.2s  eta 0.9s
 */

#ifndef PROGRESS_H
#define PROGRESS_H

#include <stddef.h>

/* ------------------------------------------------------------------------- */
/*                              Data Structures                              */
/* ------------------------------------------------------------------------- */

/**
 * @struct Progress
 * @brief State of a single progress bar.
 *
 * Treat as opaque; manipulate only through the functions below.
 */
typedef struct {
	const char *label;  /**< Short stage label, e.g. "read" */
	size_t total;       /**< Number of steps that make up 100% */
	size_t done;        /**< Steps completed so far */
	int last_pct;       /**< Last percentage drawn */
	double start;       /**< Start time in seconds (monotonic) */
	int enabled;        /**< 0 disables all output */
} Progress;

/* ------------------------------------------------------------------------- */
/*                            Public API Functions                           */
/* ------------------------------------------------------------------------- */

/**
 * @brief Initializes a progress bar and draws it at 0%.
 *
 * @param[out] p       Progress bar to initialize.
 * @param[in]  label   Stage label (not copied, must outlive the bar).
 * @param[in]  total   Number of steps (0 is treated as already complete).
 * @param[in]  enabled Non-zero to draw, zero to make every call a no-op.
 */
void progress_init(Progress *p, const char *label, size_t total, int enabled);

/**
 * @brief Advances the progress bar by @p n steps.
 *
 * @note Thread-safe; may be called concurrently from OpenMP threads.
 *
 * @param[in,out] p Progress bar.
 * @param[in]     n Number of completed steps to add.
 */
void progress_add(Progress *p, size_t n);

/**
 * @brief Advances the progress bar by one step.
 *
 * @note Thread-safe; may be called concurrently from OpenMP threads.
 *
 * @param[in,out] p Progress bar.
 */
void progress_tick(Progress *p);

/**
 * @brief Draws the final state and terminates the line.
 *
 * Must be called from outside any parallel region.
 *
 * @param[in,out] p Progress bar.
 */
void progress_finish(Progress *p);

#endif /* PROGRESS_H */
