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
 * A bar can span several stages: its label and a short note (e.g. the
 * current batch) can be changed between parallel regions, and a "unit"
 * lets the displayed count be in items while progress is counted in finer
 * steps (e.g. 4 stage steps per image).
 *
 * Output format:
 *   decode [=================>            ]  58%  580/1000  1.2s  eta 0.9s  batch 3/4
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
	char note[32];      /**< Trailing note, e.g. "batch 3/4" */
	size_t total;       /**< Number of steps that make up 100% */
	size_t unit;        /**< Steps per displayed item (>= 1) */
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
 * @param[in]  unit    Steps per displayed item (0 is treated as 1).
 * @param[in]  enabled Non-zero to draw, zero to make every call a no-op.
 */
void progress_init(Progress *p, const char *label, size_t total, size_t unit,
                   int enabled);

/**
 * @brief Changes the label and note, and redraws.
 *
 * Must be called from outside any parallel region.
 *
 * @param[in,out] p     Progress bar.
 * @param[in]     label New label (not copied), or NULL to keep the current one.
 * @param[in]     note  New note (copied, truncated), or NULL to keep the current one.
 */
void progress_set(Progress *p, const char *label, const char *note);

/**
 * @brief Erases the bar from the terminal line.
 *
 * Use before printing other messages to stderr; the next update redraws it.
 * Must be called from outside any parallel region.
 *
 * @param[in,out] p Progress bar.
 */
void progress_clear(Progress *p);

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
