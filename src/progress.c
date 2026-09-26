/**
 * @file progress.c
 * @brief Implementation of the thread-safe terminal progress bar.
 *
 * Concurrency model: the step counter is advanced with an OpenMP atomic
 * capture, so every thread knows exactly how many steps were completed
 * including its own. Drawing is serialized in a named critical section and
 * only happens when the integer percentage increases, which bounds a whole
 * run to at most ~100 redraws regardless of the number of steps.
 */

#define _POSIX_C_SOURCE 200809L

#include <omp.h>
#include <stdio.h>
#include <string.h>

#include "progress.h"

/* Width of the bar itself, in characters */
#define BAR_WIDTH 30

/* ------------------------------------------------------------------------- */
/*                            Static Helper Functions                        */
/* ------------------------------------------------------------------------- */

/**
 * @brief Renders the bar for @p done completed steps.
 *
 * Called with the "progress" critical section held (or single-threaded).
 */
static void
draw(const Progress *p, size_t done)
{
	double pct = p->total ? (double)done / (double)p->total : 1.0;
	double elapsed = omp_get_wtime() - p->start;
	int filled = (int)(pct * BAR_WIDTH);
	char bar[BAR_WIDTH + 1];

	for (int i = 0; i < BAR_WIDTH; i++) {
		if (i < filled)
			bar[i] = '=';
		else if (i == filled && done < p->total)
			bar[i] = '>';
		else
			bar[i] = ' ';
	}
	bar[BAR_WIDTH] = '\0';

	flockfile(stderr);
	fprintf(stderr, "\r%-6s [%s] %3d%%  %zu/%zu  %.1fs",
	        p->label, bar, (int)(pct * 100.0),
	        done / p->unit, p->total / p->unit, elapsed);

	if (done > 0 && done < p->total) {
		double eta = elapsed / (double)done * (double)(p->total - done);
		fprintf(stderr, "  eta %.1fs", eta);
	}

	if (p->note[0])
		fprintf(stderr, "  %s", p->note);

	/* clear leftovers from a previously longer line */
	fputs("\033[K", stderr);
	fflush(stderr);
	funlockfile(stderr);
}

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
void
progress_init(Progress *p, const char *label, size_t total, size_t unit,
              int enabled)
{
	p->label = label ? label : "";
	p->note[0] = '\0';
	p->total = total;
	p->unit = unit ? unit : 1;
	p->done = 0;
	p->last_pct = -1;
	p->start = omp_get_wtime();
	p->enabled = enabled;

	if (p->enabled) {
		p->last_pct = 0;
		draw(p, 0);
	}
}

/**
 * @brief Changes the label and note, and redraws.
 *
 * Must be called from outside any parallel region.
 *
 * @param[in,out] p     Progress bar.
 * @param[in]     label New label (not copied), or NULL to keep the current one.
 * @param[in]     note  New note (copied, truncated), or NULL to keep the current one.
 */
void
progress_set(Progress *p, const char *label, const char *note)
{
	if (label)
		p->label = label;
	if (note)
		snprintf(p->note, sizeof(p->note), "%s", note);

	if (p->enabled)
		draw(p, p->done);
}

/**
 * @brief Erases the bar from the terminal line.
 *
 * Use before printing other messages to stderr; the next update redraws it.
 * Must be called from outside any parallel region.
 *
 * @param[in,out] p Progress bar.
 */
void
progress_clear(Progress *p)
{
	if (!p->enabled)
		return;

	fputs("\r\033[K", stderr);
	fflush(stderr);
}

/**
 * @brief Advances the progress bar by @p n steps.
 *
 * @note Thread-safe; may be called concurrently from OpenMP threads.
 *
 * @param[in,out] p Progress bar.
 * @param[in]     n Number of completed steps to add.
 */
void
progress_add(Progress *p, size_t n)
{
	size_t done;
	int pct, last;

	if (!p->enabled || p->total == 0)
		return;

	#pragma omp atomic capture
	{ p->done += n; done = p->done; }

	pct = (int)(done * 100 / p->total);

	#pragma omp atomic read
	last = p->last_pct;

	if (pct <= last)
		return;

	#pragma omp critical(progress)
	{
		/* re-check: another thread may have drawn a newer state meanwhile */
		if (pct > p->last_pct) {
			#pragma omp atomic write
			p->last_pct = pct;
			draw(p, done);
		}
	}
}

/**
 * @brief Advances the progress bar by one step.
 *
 * @note Thread-safe; may be called concurrently from OpenMP threads.
 *
 * @param[in,out] p Progress bar.
 */
void
progress_tick(Progress *p)
{
	progress_add(p, 1);
}

/**
 * @brief Draws the final state and terminates the line.
 *
 * Must be called from outside any parallel region.
 *
 * @param[in,out] p Progress bar.
 */
void
progress_finish(Progress *p)
{
	if (!p->enabled)
		return;

	draw(p, p->done);
	fputc('\n', stderr);
}
