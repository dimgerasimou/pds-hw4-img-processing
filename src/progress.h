/**
 * @file progress.h
 * @brief Thread-safe terminal progress bar, drawn to stderr.
 *
 * Progress is counted in steps, displayed in items of @c unit steps (e.g.
 * several stage steps per image). The bar is redrawn only when the
 * percentage changes, so advancing it from a hot loop costs one atomic add.
 */

#ifndef PROGRESS_H
#define PROGRESS_H

#include <stddef.h>

typedef struct {
	const char *label;  /* not copied */
	char note[32];
	size_t total;
	size_t unit;
	size_t done;
	int last_pct;
	double start;
	int enabled;        /* 0: every call is a no-op */
} Progress;

void progress_init(Progress *p, const char *label, size_t total, size_t unit,
                   int enabled);

/*
 * Thread-safe, except progress_set() and progress_clear(), which must be
 * called outside parallel regions.
 */

/** @brief Changes the label and/or note (NULL keeps it) and redraws. */
void progress_set(Progress *p, const char *label, const char *note);

/** @brief Erases the bar, before printing other messages to stderr. */
void progress_clear(Progress *p);

void progress_add(Progress *p, size_t n);
void progress_tick(Progress *p);

/** @brief Draws the final state and ends the line. */
void progress_finish(Progress *p);

#endif /* PROGRESS_H */
