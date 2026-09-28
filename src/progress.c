/**
 * @file progress.c
 * @brief Terminal progress bar.
 */

#define _POSIX_C_SOURCE 200809L

#include <omp.h>
#include <stdio.h>
#include <string.h>

#include "progress.h"

#define BAR_WIDTH 30

/* Called with the "progress" critical section held, or single-threaded. */
static void
draw(const Progress *p, size_t done)
{
	if (done > p->total)
		done = p->total;

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

	/* clear leftovers of a longer previous line */
	fputs("\033[K", stderr);
	fflush(stderr);
	funlockfile(stderr);
}

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

void
progress_clear(Progress *p)
{
	if (!p->enabled)
		return;

	flockfile(stderr);
	fputs("\r\033[K", stderr);
	fflush(stderr);
	funlockfile(stderr);
}

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

void
progress_tick(Progress *p)
{
	progress_add(p, 1);
}

void
progress_finish(Progress *p)
{
	if (!p->enabled)
		return;

	draw(p, p->done);
	fputc('\n', stderr);
}
