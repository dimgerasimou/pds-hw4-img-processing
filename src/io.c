/**
 * @file io.c
 * @brief Implementation of path resolution and parallel image loading/saving.
 */

#define _POSIX_C_SOURCE 200809L

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "error.h"
#include "gpu.h"
#include "io.h"
#include "nlm.h"
#include "progress.h"

/**
 * @enum Output modes
 * @brief Where processed images are written.
 */
enum {
	OUT_NONE = 0, /**< Nothing is written */
	OUT_FILE,     /**< Single output file */
	OUT_DIR       /**< Output directory */
};

/* Stage names, indexed by STAGE_* */
static const char *stage_names[STAGE_COUNT] = {
	"read", "decode", "denoise", "edges", "encode", "write"
};

/*
 * Band height for the denoise stage. With at least one image per thread,
 * about BAND_PER_THREAD bands per thread over the batch, within
 * [BAND_MIN_ROWS, BAND_MAX_ROWS]: smaller bands balance the load better,
 * but every band recomputes 2 * patch radius extra rows of integral image,
 * so they should not get too small. With fewer images than threads, every
 * image gets a multiple of the thread count of bands of at most
 * BAND_MAX_ROWS rows, so that each image divides evenly over the threads.
 */
#define BAND_PER_THREAD 4
#define BAND_MIN_ROWS   16
#define BAND_MAX_ROWS   32

/**
 * @struct BandOrder
 * @brief Image index with its sort key, for ordering the denoise work.
 */
typedef struct {
	size_t k;           /**< Index within the batch */
	unsigned int width; /**< Image width: bands of wider images cost more */
} BandOrder;

/**
 * @struct OutName
 * @brief Output path paired with its input, for collision detection.
 */
typedef struct {
	const char *out; /**< Output path */
	const char *in;  /**< Input path it comes from */
} OutName;

/* ------------------------------------------------------------------------- */
/*                            Static Helper Functions                        */
/* ------------------------------------------------------------------------- */

/**
 * @brief Joins a directory and a file name with exactly one '/'.
 *
 * @note Caller must free.
 *
 * @return Newly allocated path, or NULL on allocation failure.
 */
static char*
path_join(const char *dir, const char *name)
{
	size_t dlen = strlen(dir);
	size_t nlen = strlen(name);
	int slash = (dlen > 0 && dir[dlen - 1] == '/');
	char *p = malloc(dlen + nlen + 2);

	if (!p) {
		DERRNOF("malloc() failed");
		return NULL;
	}

	memcpy(p, dir, dlen);
	if (!slash)
		p[dlen++] = '/';
	memcpy(p + dlen, name, nlen + 1);
	return p;
}

/**
 * @brief Returns the last component of a file path.
 */
static const char*
base_name(const char *path)
{
	const char *slash = strrchr(path, '/');
	return slash ? slash + 1 : path;
}

/**
 * @brief Checks whether a path ends in '/', i.e. explicitly names a directory.
 */
static int
has_trailing_slash(const char *path)
{
	size_t n = strlen(path);
	return n > 0 && path[n - 1] == '/';
}

/**
 * @brief Checks whether a file name has a supported image extension.
 */
static int
is_image_name(const char *name)
{
	return image_format_from_ext(name) != IMG_FMT_UNKNOWN;
}

/**
 * @brief Returns a copy of a file name with its extension replaced.
 *
 * "scan.01.png" + ".pgm" -> "scan.01.pgm"; a name without extension gets
 * @p ext appended.
 *
 * @note Caller must free.
 *
 * @return Newly allocated name, or NULL on allocation failure.
 */
static char*
replace_ext(const char *name, const char *ext)
{
	const char *dot = strrchr(name, '.');
	size_t stem = (dot && dot != name) ? (size_t)(dot - name) : strlen(name);
	size_t elen = strlen(ext);
	char *r = malloc(stem + elen + 1);

	if (!r) {
		DERRNOF("malloc() failed");
		return NULL;
	}

	memcpy(r, name, stem);
	memcpy(r + stem, ext, elen + 1);
	return r;
}

/**
 * @brief Comparison function for sorting OutName by output path.
 */
static int
cmp_outname(const void *a, const void *b)
{
	return strcmp(((const OutName *)a)->out, ((const OutName *)b)->out);
}

/**
 * @brief Refuses a set in which two inputs map to the same output path.
 *
 * @return 0 if all output paths are distinct, 1 otherwise (reported).
 */
static int
check_collisions(const ImageSet *set)
{
	OutName *v;
	int bad = 0;

	if (set->count < 2 || !set->items[0].out_path)
		return 0;

	v = malloc(set->count * sizeof(OutName));
	if (!v) {
		DERRNOF("malloc() failed");
		return 1;
	}

	for (size_t i = 0; i < set->count; i++) {
		v[i].out = set->items[i].out_path;
		v[i].in = set->items[i].in_path;
	}

	qsort(v, set->count, sizeof(OutName), cmp_outname);

	for (size_t i = 1; i < set->count; i++) {
		if (strcmp(v[i - 1].out, v[i].out) == 0) {
			uerrf("\"%s\" and \"%s\" would both be written to \"%s\"",
			      v[i - 1].in, v[i].in, v[i].out);
			bad = 1;
		}
	}

	free(v);
	return bad;
}

/**
 * @brief Checks whether two stat results refer to the same file.
 */
static int
same_file(const struct stat *a, const struct stat *b)
{
	return a->st_dev == b->st_dev && a->st_ino == b->st_ino;
}

/**
 * @brief Comparison function for sorting strings.
 */
static int
cmp_str(const void *a, const void *b)
{
	return strcmp(*(const char *const *)a, *(const char *const *)b);
}

/**
 * @brief Frees an array of strings.
 */
static void
free_names(char **names, size_t n)
{
	if (!names)
		return;

	for (size_t i = 0; i < n; i++)
		free(names[i]);
	free(names);
}

/**
 * @brief Lists the image files of a directory, sorted by name.
 *
 * Non-recursive. Hidden files and anything that is not a regular file
 * (after following symlinks) are ignored.
 *
 * @param[in]  dir   Directory to scan.
 * @param[out] names Newly allocated array of file names (caller frees).
 * @param[out] count Number of names.
 *
 * @return 0 on success, 1 on error (already reported).
 */
static int
list_dir(const char *dir, char ***names, size_t *count)
{
	DIR *d;
	struct dirent *ent;
	char **v = NULL;
	size_t n = 0, cap = 0;

	*names = NULL;
	*count = 0;

	d = opendir(dir);
	if (!d) {
		uerrnof(errno, "cannot open directory \"%s\"", dir);
		return 1;
	}

	errno = 0;
	while ((ent = readdir(d)) != NULL) {
		struct stat st;
		char *full;
		int ok;

		if (ent->d_name[0] == '.' || !is_image_name(ent->d_name))
			continue;

		full = path_join(dir, ent->d_name);
		if (!full)
			goto fail;
		ok = (stat(full, &st) == 0 && S_ISREG(st.st_mode));
		free(full);

		if (!ok)
			continue;

		if (n == cap) {
			size_t ncap = cap ? cap * 2 : 64;
			char **nv = realloc(v, ncap * sizeof(char *));
			if (!nv) {
				DERRNOF("realloc() failed");
				goto fail;
			}
			v = nv;
			cap = ncap;
		}

		v[n] = strdup(ent->d_name);
		if (!v[n]) {
			DERRNOF("strdup() failed");
			goto fail;
		}
		n++;
		errno = 0;
	}

	if (errno != 0) {
		uerrnof(errno, "cannot read directory \"%s\"", dir);
		goto fail;
	}

	closedir(d);

	if (n > 1)
		qsort(v, n, sizeof(char *), cmp_str);

	*names = v;
	*count = n;
	return 0;

fail:
	closedir(d);
	free_names(v, n);
	return 1;
}

/**
 * @brief Decides the output mode.
 *
 * Nothing is created here: when the output directory does not exist yet,
 * @p create is set and the caller creates it once every other check has
 * passed, so a refused run leaves no empty directory behind. Refuses to
 * overwrite the input.
 *
 * @param[in]  input  Input path.
 * @param[in]  in_st  stat() of the input.
 * @param[in]  output Output path, or NULL.
 * @param[out] mode   OUT_NONE, OUT_FILE or OUT_DIR.
 * @param[out] create 1 if the output directory must be created.
 *
 * @return 0 on success, 1 on error (already reported).
 */
static int
resolve_output(const char *input, const struct stat *in_st,
               const char *output, int *mode, int *create)
{
	int in_dir = S_ISDIR(in_st->st_mode);
	struct stat out_st;

	*mode = OUT_NONE;
	*create = 0;

	if (!output)
		return 0;

	if (stat(output, &out_st) == 0) {
		if (S_ISDIR(out_st.st_mode)) {
			*mode = OUT_DIR;
		} else if (in_dir) {
			uerrf("cannot write a directory of images into file \"%s\"", output);
			return 1;
		} else if (S_ISREG(out_st.st_mode)) {
			*mode = OUT_FILE;
		} else {
			uerrf("\"%s\" is not a regular file or directory", output);
			return 1;
		}

		if (same_file(in_st, &out_st)) {
			if (in_dir)
				uerrf("output directory \"%s\" is the input directory", output);
			else
				uerrf("\"%s\" and \"%s\" are the same file", input, output);
			return 1;
		}

		return 0;
	}

	if (errno != ENOENT) {
		uerrnof(errno, "cannot access \"%s\"", output);
		return 1;
	}

	if (in_dir || has_trailing_slash(output)) {
		*mode = OUT_DIR;
		*create = 1;
	} else {
		*mode = OUT_FILE;
	}

	return 0;
}

/**
 * @brief Picks the output format.
 *
 * Explicit request first; then, for a single output file, the extension of
 * that file if it names a writable format; PGM otherwise.
 */
static int
choose_format(int requested, int mode, const char *output)
{
	if (requested != IMG_FMT_UNKNOWN)
		return requested;

	if (mode == OUT_FILE) {
		int f = image_format_from_ext(output);
		if (image_format_writable(f))
			return f;
	}

	return IMG_FMT_PGM;
}

/**
 * @brief Reads a whole file into a newly allocated buffer.
 *
 * @param[in]  path Path to read.
 * @param[out] buf  Newly allocated contents (caller frees).
 * @param[out] len  Size in bytes.
 * @param[out] err  errno on failure.
 *
 * @return IMG_OK, IMG_ERR_SYS, IMG_ERR_NOMEM or IMG_ERR_TRUNC.
 */
static int
file_read(const char *path, unsigned char **buf, size_t *len, int *err)
{
	struct stat st;
	unsigned char *p;
	size_t size, done = 0;
	int fd;

	*buf = NULL;
	*len = 0;

	fd = open(path, O_RDONLY);
	if (fd < 0) {
		*err = errno;
		return IMG_ERR_SYS;
	}

	if (fstat(fd, &st) != 0) {
		*err = errno;
		close(fd);
		return IMG_ERR_SYS;
	}

	size = (size_t)st.st_size;
	if (size == 0) {
		close(fd);
		return IMG_ERR_TRUNC;
	}

	p = malloc(size);
	if (!p) {
		close(fd);
		return IMG_ERR_NOMEM;
	}

	while (done < size) {
		ssize_t r = read(fd, p + done, size - done);

		if (r > 0) {
			done += (size_t)r;
			continue;
		}
		if (r < 0 && errno == EINTR)
			continue;

		/* r == 0: file shrank while reading */
		*err = (r < 0) ? errno : 0;
		free(p);
		close(fd);
		return (r < 0) ? IMG_ERR_SYS : IMG_ERR_TRUNC;
	}

	close(fd);
	*buf = p;
	*len = size;
	return IMG_OK;
}

/**
 * @brief Writes a buffer to a file, creating or truncating it.
 *
 * @param[in]  path Path to write.
 * @param[in]  buf  Data.
 * @param[in]  len  Size in bytes.
 * @param[out] err  errno on failure.
 *
 * @return IMG_OK or IMG_ERR_SYS.
 */
static int
file_write(const char *path, const unsigned char *buf, size_t len, int *err)
{
	size_t done = 0;
	int fd;

	fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
	if (fd < 0) {
		*err = errno;
		return IMG_ERR_SYS;
	}

	while (done < len) {
		ssize_t w = write(fd, buf + done, len - done);

		if (w > 0) {
			done += (size_t)w;
			continue;
		}
		if (w < 0 && errno == EINTR)
			continue;

		*err = errno;
		close(fd);
		return IMG_ERR_SYS;
	}

	/* delayed write errors (e.g. on NFS) surface at close() */
	if (close(fd) != 0) {
		*err = errno;
		return IMG_ERR_SYS;
	}

	return IMG_OK;
}

/**
 * @brief Read stage for one image: file -> buf.
 */
static int
do_read(ImageItem *it)
{
	int r = file_read(it->in_path, &it->buf, &it->buf_len, &it->sys_errno[STAGE_READ]);
	it->bytes[STAGE_READ] = it->buf_len;
	return r;
}

/**
 * @brief Decode stage for one image: buf -> img. Releases buf.
 */
static int
do_decode(ImageItem *it)
{
	int r = image_decode(it->buf, it->buf_len, &it->img, &it->in_format, &it->detail);

	free(it->buf);
	it->buf = NULL;
	it->buf_len = 0;

	it->width = it->img.width;
	it->height = it->img.height;
	it->bytes[STAGE_DECODE] = image_bytes(&it->img);
	return r;
}

/**
 * @brief Encode stage for one image: img -> buf. Releases img.
 *
 * Encoding is the last use of the pixels, so they are freed right away.
 */
static int
do_encode(ImageItem *it, int fmt)
{
	int r;

	it->bytes[STAGE_ENCODE] = image_bytes(&it->img);
	r = image_encode(&it->img, fmt, &it->buf, &it->buf_len);
	image_free(&it->img);
	return r;
}

/**
 * @brief Write stage for one image: buf -> file. Releases buf.
 */
static int
do_write(ImageItem *it)
{
	int r = file_write(it->out_path, it->buf, it->buf_len, &it->sys_errno[STAGE_WRITE]);

	it->bytes[STAGE_WRITE] = (r == IMG_OK) ? it->buf_len : 0;

	free(it->buf);
	it->buf = NULL;
	it->buf_len = 0;
	return r;
}

/**
 * @brief Progress units of one image in a stage.
 *
 * Denoising costs about as many units as its search window has offsets;
 * every other stage counts 1. See io_progress_units().
 */
static size_t
io_stage_units(const ImageSet *set, int stage)
{
	return (stage == STAGE_DENOISE) ? nlm_units(&set->denoise.params) : 1;
}

/**
 * @brief Adds @p wall seconds to a stage's total and marks it performed.
 *
 * Atomic: with the GPU pipeline, the CPU and GPU threads can finish parts
 * of the same stage (preparation and GPU run of denoising) concurrently.
 */
static void
stage_done(ImageSet *set, int stage, double wall)
{
	#pragma omp atomic
	set->wall_time_s[stage] += wall;

	#pragma omp atomic write
	set->performed[stage] = 1;
}

/**
 * @brief Marks a stage as not attempted for an image.
 *
 * @param[in,out] it    Image.
 * @param[in]     stage STAGE_* value.
 * @param[in]     units Progress units of the stage for other stages.
 *
 * @return Progress units the skipped image still accounts for.
 */
static size_t
skip(ImageItem *it, int stage, size_t units)
{
	/* also tells the compiler the range, which the OpenMP-outlined caller loses */
	if (stage < 0 || stage >= STAGE_COUNT)
		return units;

	it->err[stage] = IO_NOT_DONE;
	return (stage == STAGE_DENOISE) ? it->units : units;
}

/**
 * @brief Checks whether an image holds valid pixels after the pixel stages.
 *
 * Denoising and edge detection are optional: the pixels come from the last
 * of decode, denoise and edges that is enabled.
 */
static int
pixels_ok(const ImageSet *set, const ImageItem *it)
{
	if (set->edges.enabled)
		return it->err[STAGE_EDGES] == IMG_OK;
	if (set->denoise.enabled)
		return it->err[STAGE_DENOISE] == IMG_OK;
	return it->err[STAGE_DECODE] == IMG_OK;
}

/**
 * @brief Checks whether an image holds valid pixels for edge detection:
 *        denoised if denoising is enabled, decoded otherwise.
 */
static int
edges_input_ok(const ImageSet *set, const ImageItem *it)
{
	if (set->denoise.enabled)
		return it->err[STAGE_DENOISE] == IMG_OK;
	return it->err[STAGE_DECODE] == IMG_OK;
}

/**
 * @brief Checks whether an image should take part in a stage.
 */
static int
eligible(const ImageSet *set, const ImageItem *it, int stage)
{
	switch (stage) {
	case STAGE_READ:    return 1;
	case STAGE_DECODE:  return it->err[STAGE_READ] == IMG_OK;
	case STAGE_DENOISE: return set->denoise.enabled && it->err[STAGE_DECODE] == IMG_OK;
	case STAGE_EDGES:   return set->edges.enabled && edges_input_ok(set, it);
	case STAGE_ENCODE:  return it->out_path && pixels_ok(set, it);
	case STAGE_WRITE:   return it->err[STAGE_ENCODE] == IMG_OK;
	default:            return 0;
	}
}

/**
 * @brief Distributes the denoise progress units of a batch over its images.
 *
 * The batch owns (images x nlm_units()) units, a constant, so the bar's
 * total never changes. Within the batch they are shared in proportion to
 * the images' pixel counts, since denoising time is proportional to them:
 * without this, bands of a large image would advance the bar as much as
 * those of a small one. Images that will not be denoised keep one image's
 * worth, which they add at once. Shares are rounded cumulatively so that
 * they add up exactly.
 */
static void
assign_units(ImageSet *set, size_t first, size_t last)
{
	const size_t per = nlm_units(&set->denoise.params);
	size_t pool = 0, done = 0;
	unsigned long long px = 0, acc = 0;

	for (size_t i = first; i < last; i++) {
		ImageItem *it = &set->items[i];

		if (eligible(set, it, STAGE_DENOISE)) {
			px += image_bytes(&it->img);
			pool += per;
		} else {
			it->units = per;
		}
	}

	for (size_t i = first; i < last; i++) {
		ImageItem *it = &set->items[i];
		size_t upto;

		if (!eligible(set, it, STAGE_DENOISE))
			continue;

		acc += image_bytes(&it->img);
		upto = px ? (size_t)((unsigned long long)pool * acc / px) : pool;
		it->units = upto - done;
		done = upto;
	}
}

/**
 * @brief Comparison function: wider images first.
 */
static int
cmp_band_order(const void *a, const void *b)
{
	const BandOrder *x = a, *y = b;

	if (x->width != y->width)
		return (x->width < y->width) ? 1 : -1;
	return (x->k > y->k) - (x->k < y->k);
}

/**
 * @brief Finds the job owning global band @p g: start[k] <= g < start[k+1].
 */
static size_t
find_job(const long *start, size_t n, long g)
{
	size_t lo = 0, hi = n;

	/* last k with start[k] <= g */
	while (hi - lo > 1) {
		size_t mid = lo + (hi - lo) / 2;
		if (start[mid] <= g)
			lo = mid;
		else
			hi = mid;
	}

	return lo;
}

/**
 * @brief Band height for an image of @p h rows (see BAND_* above).
 *
 * @param[in] h       Image height.
 * @param[in] band    Band height used with at least one image per thread.
 * @param[in] few     Non-zero when the batch has fewer images than threads.
 * @param[in] threads Number of threads.
 */
static long
band_rows(long h, long band, int few, long threads)
{
	long nb;

	if (!few)
		return band;

	nb = (h + BAND_MAX_ROWS - 1) / BAND_MAX_ROWS;
	nb = ((nb + threads - 1) / threads) * threads;
	band = (h + nb - 1) / nb;
	return band > 0 ? band : 1;
}

/**
 * @brief Denoise stage over the images [first, last).
 *
 * 1. Prepare every image (noise estimate, padding, weight table): with at
 *    least one image per thread, one image per thread; otherwise one image
 *    at a time, each using all threads.
 * 2. Run every band of every image as one pool of work, handed out to the
 *    threads dynamically: a large image is spread over all threads and the
 *    idle time at the end is at most about one band. Bands of wider (more
 *    expensive) images are handed out first, so the last ones are cheap.
 * 3. Collect the outputs.
 *
 * An image's recorded time is the thread time spent on it (preparation plus
 * its bands).
 *
 * @return Number of images that failed.
 */
static size_t
denoise_bands(ImageSet *set, size_t first, size_t last, Progress *progress)
{
	const NlmParams *p = &set->denoise.params;
	const size_t n = last - first;
	const long threads = omp_get_max_threads();
	NlmJob *jobs = calloc(n ? n : 1, sizeof(NlmJob));
	BandOrder *order = malloc((n ? n : 1) * sizeof(BandOrder));
	long *start = malloc((n + 1) * sizeof(long));
	long *left = calloc(n ? n : 1, sizeof(long));
	unsigned int maxw = 0;
	long rows = 0, band, maxband = 0, total;
	size_t failed = 0, m = 0, count = 0;
	int nomem = 0, few;

	if (!jobs || !order || !start || !left) {
		DERRNOF("allocation failed");
		free(jobs); free(order); free(start); free(left);
		for (size_t i = first; i < last; i++) {
			ImageItem *it = &set->items[i];
			it->err[STAGE_DENOISE] = eligible(set, it, STAGE_DENOISE) ? IMG_ERR_NOMEM : IO_NOT_DONE;
			failed += (it->err[STAGE_DENOISE] == IMG_ERR_NOMEM);
			progress_add(progress, it->units);
		}
		return failed;
	}

	for (size_t k = 0; k < n; k++) {
		const ImageItem *it = &set->items[first + k];
		if (eligible(set, it, STAGE_DENOISE)) {
			rows += (long)it->img.height;
			if (it->img.width > maxw)
				maxw = it->img.width;
			count++;
		}
	}

	few = (count < (size_t)threads);
	band = rows / (BAND_PER_THREAD * threads);
	band = band < BAND_MIN_ROWS ? BAND_MIN_ROWS : band > BAND_MAX_ROWS ? BAND_MAX_ROWS : band;

	/*
	 * 1. prepare: with few images, each one with all threads (noise estimate,
	 *    padding and table are parallel inside); otherwise one per thread
	 */
	#pragma omp parallel for schedule(dynamic) reduction(+:failed) if(!few)
	for (size_t k = 0; k < n; k++) {
		ImageItem *it = &set->items[first + k];
		double ts;

		if (!eligible(set, it, STAGE_DENOISE)) {
			it->err[STAGE_DENOISE] = IO_NOT_DONE;
			progress_add(progress, it->units);
			continue;
		}

		it->bytes[STAGE_DENOISE] = image_bytes(&it->img);
		ts = omp_get_wtime();
		it->err[STAGE_DENOISE] = nlm_job_prepare(&it->img, p,
		                                         band_rows((long)it->img.height, band, few, threads),
		                                         few, &jobs[k], &it->sigma);
		it->time_s[STAGE_DENOISE] = omp_get_wtime() - ts;

		if (it->err[STAGE_DENOISE] != IMG_OK) {
			failed++;
			progress_add(progress, it->units);
		}
	}

	for (size_t k = 0; k < n; k++)
		if (set->items[first + k].err[STAGE_DENOISE] == IMG_OK && jobs[k].band > maxband)
			maxband = jobs[k].band;

	/* global band numbering, wider images first */
	for (size_t k = 0; k < n; k++)
		if (set->items[first + k].err[STAGE_DENOISE] == IMG_OK) {
			order[m].k = k;
			order[m].width = set->items[first + k].img.width;
			m++;
		}

	qsort(order, m, sizeof(BandOrder), cmp_band_order);

	start[0] = 0;
	for (size_t j = 0; j < m; j++) {
		left[j] = nlm_job_bands(&jobs[order[j].k]);
		start[j + 1] = start[j] + left[j];
	}
	total = start[m];

	/* 2. all bands of all images */
	#pragma omp parallel
	{
		NlmScratch *s = NULL;

		if (total > 0) {
			s = nlm_scratch_new(maxw, maxband, p->patch);
			if (!s) {
				#pragma omp atomic write
				nomem = 1;
			}
		}

		#pragma omp for schedule(dynamic)
		for (long g = 0; g < total; g++) {
			size_t j = find_job(start, m, g);
			size_t k = order[j].k;
			double ts, dt;

			if (!s)
				continue;

			ts = omp_get_wtime();
			nlm_job_run(&jobs[k], g - start[j], s);
			dt = omp_get_wtime() - ts;

			#pragma omp atomic
			set->items[first + k].time_s[STAGE_DENOISE] += dt;

			#pragma omp atomic update
			left[j]--;

			progress_add(progress, nlm_band_units(set->items[first + k].units,
			                                      g - start[j], nlm_job_bands(&jobs[k])));
		}

		nlm_scratch_free(s);
	}

	/* 3. collect */
	for (size_t j = 0; j < m; j++) {
		size_t k = order[j].k;
		ImageItem *it = &set->items[first + k];
		Image out;

		/* done at preparation (nothing to run) */
		if (nlm_job_bands(&jobs[k]) == 0)
			progress_add(progress, it->units);

		/* bands never run (no memory): account for them so the bar completes */
		if (nomem && left[j] > 0)
			for (long b = nlm_job_bands(&jobs[k]) - left[j]; b < nlm_job_bands(&jobs[k]); b++)
				progress_add(progress, nlm_band_units(it->units, b, nlm_job_bands(&jobs[k])));

		if (nomem) {
			nlm_job_discard(&jobs[k]);
			it->err[STAGE_DENOISE] = IMG_ERR_NOMEM;
			failed++;
			continue;
		}

		nlm_job_finish(&jobs[k], &out);
		image_free(&it->img);
		it->img = out;
	}

	free(jobs);
	free(order);
	free(start);
	free(left);
	return failed;
}

/**
 * @brief Reports the failures of a stage within a range, in input order.
 */
static void
report_failures(const ImageSet *set, int stage, size_t first, size_t last)
{
	for (size_t i = first; i < last; i++) {
		const ImageItem *it = &set->items[i];
		int e = it->err[stage];

		if (e == IMG_OK || e == IO_NOT_DONE)
			continue;

		switch (stage) {
		case STAGE_READ:
			uerrf("cannot read \"%s\": %s", it->in_path,
			      image_strerror(e, it->sys_errno[stage]));
			break;
		case STAGE_DENOISE:
			uerrf("cannot denoise \"%s\": %s", it->in_path,
			      image_strerror(e, 0));
			break;
		case STAGE_EDGES:
			uerrf("cannot detect edges of \"%s\": %s", it->in_path,
			      image_strerror(e, 0));
			break;
		case STAGE_DECODE:
			if (it->detail)
				uerrf("cannot decode \"%s\": %s (%s)", it->in_path,
				      image_strerror(e, 0), it->detail);
			else
				uerrf("cannot decode \"%s\": %s", it->in_path,
				      image_strerror(e, 0));
			break;
		case STAGE_ENCODE:
			uerrf("cannot encode \"%s\": %s", it->out_path,
			      image_strerror(e, 0));
			break;
		case STAGE_WRITE:
			uerrf("cannot write \"%s\": %s", it->out_path,
			      image_strerror(e, it->sys_errno[stage]));
			break;
		}
	}
}

/* ------------------------------------------------------------------------- */
/*                            Public API Functions                           */
/* ------------------------------------------------------------------------- */

/**
 * @brief Builds the image set from the input and output arguments.
 *
 * Resolves the input (file or directory), lists the images, decides the
 * output format and where each image is written, checks for name
 * collisions, and creates the output directory if needed. No image data is
 * read.
 *
 * The output format is @p format if given; otherwise, for a single output
 * file with a writable extension (.pgm/.png), the format of that extension;
 * otherwise PGM.
 *
 * @param[in]  input  Input file or directory.
 * @param[in]  output Output file or directory, or NULL to not write.
 * @param[in]  format Requested output format, or IMG_FMT_UNKNOWN for automatic.
 * @param[out] set    Image set to fill (zeroed on failure).
 *
 * @return 0 on success, 1 on error (already reported).
 */
int
io_resolve(const char *input, const char *output, int format, ImageSet *set)
{
	struct stat in_st;
	char **names = NULL;
	size_t n = 0;
	int in_dir, mode, create;

	memset(set, 0, sizeof(*set));

	if (stat(input, &in_st) != 0) {
		uerrnof(errno, "cannot access \"%s\"", input);
		return 1;
	}

	in_dir = S_ISDIR(in_st.st_mode);
	if (!in_dir && !S_ISREG(in_st.st_mode)) {
		uerrf("\"%s\" is not a regular file or directory", input);
		return 1;
	}

	if (resolve_output(input, &in_st, output, &mode, &create))
		return 1;

	set->out_format = (mode == OUT_NONE) ? IMG_FMT_UNKNOWN
	                                     : choose_format(format, mode, output);

	/* an explicit -f that contradicts the -o file name is almost always a typo */
	if (mode == OUT_FILE && format != IMG_FMT_UNKNOWN) {
		int ext = image_format_from_ext(output);
		if (ext != IMG_FMT_UNKNOWN && ext != set->out_format)
			uerrf("warning: writing %s data to \"%s\"",
			      image_format_name(set->out_format), output);
	}

	if (in_dir) {
		if (list_dir(input, &names, &n))
			return 1;
		if (n == 0) {
			uerrf("no supported images (.pgm/.pnm/.png/.jpg/.jpeg/.bmp/.tga) found in \"%s\"",
			      input);
			return 1;
		}
	} else {
		n = 1;
	}

	set->items = calloc(n, sizeof(ImageItem));
	if (!set->items) {
		DERRNOF("calloc() failed");
		free_names(names, n);
		return 1;
	}
	set->count = n;

	for (size_t i = 0; i < n; i++) {
		ImageItem *it = &set->items[i];
		const char *name = in_dir ? names[i] : base_name(input);

		for (int s = 0; s < STAGE_COUNT; s++)
			it->err[s] = IO_NOT_DONE;

		it->in_path = in_dir ? path_join(input, name) : strdup(input);
		if (!it->in_path)
			goto fail;

		if (mode == OUT_FILE) {
			it->out_path = strdup(output);
		} else if (mode == OUT_DIR) {
			char *out_name = replace_ext(name, image_format_ext(set->out_format));
			if (!out_name)
				goto fail;
			it->out_path = path_join(output, out_name);
			free(out_name);
		}

		if (mode != OUT_NONE && !it->out_path)
			goto fail;
	}

	if (mode == OUT_DIR && check_collisions(set))
		goto fail_quiet;

	/* file -> dir: the destination may still turn out to be the input itself */
	if (!in_dir && mode == OUT_DIR && !create) {
		struct stat dst;
		if (stat(set->items[0].out_path, &dst) == 0 && same_file(&in_st, &dst)) {
			uerrf("\"%s\" and \"%s\" are the same file",
			      input, set->items[0].out_path);
			goto fail_quiet;
		}
	}

	/* every check passed: only now create the output directory */
	if (create && mkdir(output, 0755) != 0) {
		uerrnof(errno, "cannot create directory \"%s\"", output);
		goto fail_quiet;
	}

	free_names(names, n);
	return 0;

fail:
	DERRNOF("failed to build path list");
fail_quiet:
	free_names(names, n);
	io_free(set);
	return 1;
}

/*
 * Band height for edge detection: about EDGE_BANDS_PER_THREAD bands per
 * thread over the batch, within [EDGE_MIN_ROWS, EDGE_MAX_ROWS]. Every band
 * recomputes 4 + 2 * (Gaussian radius) border rows of blur, so bands should
 * not be too small.
 */
#define EDGE_BANDS_PER_THREAD 4
#define EDGE_MIN_ROWS         16
#define EDGE_MAX_ROWS         64

/**
 * @brief Edge detection stage over the images [first, last).
 *
 * Like denoising: the bands of all images of the batch form one pool,
 * handed out to the threads dynamically, bands of wider images first. Each
 * band is classified with the thread's own small buffers (canny_band()).
 * The thread that completes the last band of an image runs its hysteresis
 * right away, so hysteresis overlaps with the other images' bands instead
 * of forming a separate phase. The class map is a separate buffer, since
 * other bands still read the image; it replaces the image once complete.
 *
 * An image's recorded time is the thread time spent on it (its bands plus
 * its hysteresis).
 *
 * @return Number of images that failed.
 */
static size_t
edges_bands(ImageSet *set, size_t first, size_t last, Progress *progress)
{
	const CannySetup *cs = &set->edges.setup;
	const size_t n = last - first;
	const long threads = omp_get_max_threads();
	unsigned char **maps = calloc(n ? n : 1, sizeof(unsigned char *));
	BandOrder *order = malloc((n ? n : 1) * sizeof(BandOrder));
	long *start = malloc((n + 1) * sizeof(long));
	long *left = calloc(n ? n : 1, sizeof(long));
	long rows = 0, band, total;
	unsigned int maxw = 0;
	size_t failed = 0, m = 0;
	int nomem = 0;

	if (!maps || !order || !start || !left) {
		DERRNOF("allocation failed");
		free(maps); free(order); free(start); free(left);
		for (size_t i = first; i < last; i++) {
			ImageItem *it = &set->items[i];
			it->err[STAGE_EDGES] = eligible(set, it, STAGE_EDGES) ? IMG_ERR_NOMEM : IO_NOT_DONE;
			failed += (it->err[STAGE_EDGES] == IMG_ERR_NOMEM);
			progress_add(progress, 1);
		}
		return failed;
	}

	/* eligible images, their class maps, and the band height */
	for (size_t k = 0; k < n; k++) {
		ImageItem *it = &set->items[first + k];

		if (!eligible(set, it, STAGE_EDGES)) {
			it->err[STAGE_EDGES] = IO_NOT_DONE;
			progress_add(progress, 1);
			continue;
		}

		it->bytes[STAGE_EDGES] = image_bytes(&it->img);
		it->time_s[STAGE_EDGES] = 0.0;
		maps[k] = malloc(image_bytes(&it->img));
		if (!maps[k]) {
			it->err[STAGE_EDGES] = IMG_ERR_NOMEM;
			failed++;
			progress_add(progress, 1);
			continue;
		}

		it->err[STAGE_EDGES] = IO_NOT_DONE; /* until its hysteresis is done */
		rows += (long)it->img.height;
		if (it->img.width > maxw)
			maxw = it->img.width;
		order[m].k = k;
		order[m].width = it->img.width;
		m++;
	}

	band = rows / (EDGE_BANDS_PER_THREAD * threads);
	band = band < EDGE_MIN_ROWS ? EDGE_MIN_ROWS : band > EDGE_MAX_ROWS ? EDGE_MAX_ROWS : band;

	/* global band numbering, wider images first */
	qsort(order, m, sizeof(BandOrder), cmp_band_order);

	start[0] = 0;
	for (size_t j = 0; j < m; j++) {
		const ImageItem *it = &set->items[first + order[j].k];
		left[j] = ((long)it->img.height + band - 1) / band;
		start[j + 1] = start[j] + left[j];
	}
	total = start[m];

	#pragma omp parallel reduction(+:failed)
	{
		CannyScratch *sc = total > 0 ? canny_scratch_new((int)maxw, (int)band, cs->radius) : NULL;

		if (total > 0 && !sc) {
			#pragma omp atomic write
			nomem = 1;
		}

		#pragma omp for schedule(dynamic)
		for (long g = 0; g < total; g++) {
			const size_t j = find_job(start, m, g);
			const size_t k = order[j].k;
			ImageItem *it = &set->items[first + k];
			const long y0 = (g - start[j]) * band;
			const long y1 = (y0 + band < (long)it->img.height) ? y0 + band : (long)it->img.height;
			double ts = omp_get_wtime(), dt;
			long rem;

			if (!sc)
				continue;

			canny_band(&it->img, cs, (int)y0, (int)y1, maps[k], sc);

			#pragma omp atomic capture
			rem = --left[j];

			/* last band of the image: hysteresis, then the image becomes the edge map */
			if (rem == 0) {
				int r = canny_hysteresis(maps[k], (int)it->img.width, (int)it->img.height);

				if (r == IMG_OK) {
					free(it->img.data);
					it->img.data = maps[k];
					maps[k] = NULL;
				} else {
					failed++;
				}
				it->err[STAGE_EDGES] = r;
				progress_add(progress, 1);
			}

			dt = omp_get_wtime() - ts;
			#pragma omp atomic
			it->time_s[STAGE_EDGES] += dt;
		}

		canny_scratch_free(sc);
	}

	/* images whose bands never ran (no scratch memory) */
	for (size_t j = 0; j < m; j++) {
		ImageItem *it = &set->items[first + order[j].k];

		if (left[j] > 0 && nomem) {
			it->err[STAGE_EDGES] = IMG_ERR_NOMEM;
			failed++;
			progress_add(progress, 1);
		}
	}

	for (size_t k = 0; k < n; k++)
		free(maps[k]);
	free(maps);
	free(order);
	free(start);
	free(left);
	return failed;
}

/**
 * @brief Runs one stage over the eligible images of a range, in parallel.
 *
 * An image is eligible if it passed the previous stage (and, for encode and
 * write, has an output path). Failures are reported after the parallel
 * region, in input order, unless set->quiet is set. The wall time of the
 * call is added to the stage's total in the set.
 *
 * Every image of the range advances @p progress by its units for the
 * stage, eligible or not, so a bar sized with io_progress_units() always
 * reaches 100%. The denoise stage advances it band by band.
 *
 * Data volume recorded per image: bytes read (read), pixel bytes produced
 * (decode), pixel bytes consumed (encode), bytes written (write).
 *
 * Buffers are released as soon as they are no longer needed: file contents
 * after decode, pixels after encode, encoded data after write.
 *
 * @param[in,out] set      Image set.
 * @param[in]     stage    STAGE_* value.
 * @param[in]     first    First image of the range.
 * @param[in]     last     One past the last image of the range.
 * @param[in,out] progress Progress bar to advance (may be disabled).
 *
 * @return Number of images that failed this stage.
 */
size_t
io_run(ImageSet *set, int stage, size_t first, size_t last, Progress *progress)
{
	size_t failed = 0, skip_units;
	double t0;

	if (stage < 0 || stage >= STAGE_COUNT) {
		DERRF("invalid stage: %d", stage);
		return 0;
	}

	if (last > set->count)
		last = set->count;

	t0 = omp_get_wtime();

	if (stage == STAGE_DENOISE) {
		assign_units(set, first, last);
		failed = denoise_bands(set, first, last, progress);
		goto done;
	}

	if (stage == STAGE_EDGES) {
		failed = edges_bands(set, first, last, progress);
		goto done;
	}

	skip_units = io_stage_units(set, stage);

	#pragma omp parallel for schedule(dynamic) reduction(+:failed)
	for (size_t i = first; i < last; i++) {
		ImageItem *it = &set->items[i];
		double ts;
		int r;

		if (!eligible(set, it, stage)) {
			progress_add(progress, skip(it, stage, skip_units));
			continue;
		}

		ts = omp_get_wtime();
		switch (stage) {
		case STAGE_READ:   r = do_read(it);                     break;
		case STAGE_DECODE:  r = do_decode(it);                           break;
		case STAGE_ENCODE:  r = do_encode(it, set->out_format);          break;
		default:            r = do_write(it);                            break;
		}
		it->time_s[stage] = omp_get_wtime() - ts;
		it->err[stage] = r;

		if (r != IMG_OK)
			failed++;

		progress_tick(progress);
	}

done:
	stage_done(set, stage, omp_get_wtime() - t0);

	/* move the bar out of the way so the messages get their own lines */
	if (failed && !set->quiet) {
		progress_clear(progress);
		report_failures(set, stage, first, last);
	}

	return failed;
}

/**
 * @brief GPU denoising, CPU part: prepares the images of a range.
 *
 * Computes, for every image, the noise estimate, the mirrored padding and
 * the weight table (nlm_job_prepare()), in parallel over the images (or,
 * with fewer images than threads, one image at a time with all threads).
 * The prepared jobs are kept in the items for io_gpu_filter().
 *
 * @param[in,out] set      Image set (denoising must be enabled).
 * @param[in]     first    First image of the range.
 * @param[in]     last     One past the last image of the range.
 * @param[in,out] progress Progress bar (advanced for images that are skipped).
 *
 * @return Number of images that failed.
 */
size_t
io_prepare(ImageSet *set, size_t first, size_t last, Progress *progress)
{
	const NlmParams *p = &set->denoise.params;
	size_t failed = 0, count = 0;
	double t0 = omp_get_wtime();
	int few;

	if (last > set->count)
		last = set->count;

	assign_units(set, first, last);

	for (size_t i = first; i < last; i++)
		count += eligible(set, &set->items[i], STAGE_DENOISE);
	few = (count < (size_t)omp_get_max_threads());

	#pragma omp parallel for schedule(dynamic) reduction(+:failed) if(!few)
	for (size_t i = first; i < last; i++) {
		ImageItem *it = &set->items[i];
		double ts;

		if (!eligible(set, it, STAGE_DENOISE)) {
			it->err[STAGE_DENOISE] = IO_NOT_DONE;
			progress_add(progress, it->units);
			continue;
		}

		/* the GPU processes the whole image at once: one band */
		it->bytes[STAGE_DENOISE] = image_bytes(&it->img);
		ts = omp_get_wtime();
		it->err[STAGE_DENOISE] = nlm_job_prepare(&it->img, p, (long)it->img.height, few,
		                                         &it->job, &it->sigma);
		it->time_s[STAGE_DENOISE] = omp_get_wtime() - ts;

		if (it->err[STAGE_DENOISE] != IMG_OK) {
			failed++;
			progress_add(progress, it->units);
		}
	}

	stage_done(set, STAGE_DENOISE, omp_get_wtime() - t0);
	return failed;
}

/**
 * @brief Denoises a prepared job on the CPU, one band after the other.
 *
 * Fallback for jobs the GPU does not support.
 */
static int
cpu_fallback(NlmJob *job, unsigned int width, unsigned int patch)
{
	NlmScratch *s = nlm_scratch_new(width, job->band, patch);

	if (!s)
		return IMG_ERR_NOMEM;

	for (long b = 0; b < nlm_job_bands(job); b++)
		nlm_job_run(job, b, s);

	nlm_scratch_free(s);
	return IMG_OK;
}

/**
 * @brief GPU filters: denoises and/or detects the edges of a range.
 *
 * Runs every image through the enabled filters on the GPU, one image after
 * the other: denoising for the images prepared by io_prepare(), then edge
 * detection. An image whose denoising parameters exceed the GPU's limits is
 * denoised on the CPU instead. Meant to run on its own thread while the CPU
 * stages work on other batches.
 *
 * @param[in,out] set      Image set.
 * @param[in]     first    First image of the range.
 * @param[in]     last     One past the last image of the range.
 * @param[in,out] progress Progress bar (advanced per image and filter).
 *
 * @return Number of stage failures.
 */
size_t
io_gpu_filter(ImageSet *set, size_t first, size_t last, Progress *progress)
{
	size_t failed_dn = 0, failed_ed = 0;
	double wall_dn = 0.0, wall_ed = 0.0;

	if (last > set->count)
		last = set->count;

	for (size_t i = first; i < last; i++) {
		ImageItem *it = &set->items[i];
		int on_gpu = 0;
		double ts;
		Image out;
		int r;

		/* denoising: images prepared by io_prepare() (others already accounted for) */
		if (set->denoise.enabled && it->err[STAGE_DENOISE] == IMG_OK) {
			ts = omp_get_wtime();

			/* followed by edge detection: keep the result on the GPU */
			r = gpu_denoise(&it->job, &set->gpu_time, set->edges.enabled);
			on_gpu = (r == IMG_OK && set->edges.enabled);
			if (r == IMG_ERR_UNSUPPORTED)
				r = cpu_fallback(&it->job, it->img.width, set->denoise.params.patch);

			if (r == IMG_OK) {
				nlm_job_finish(&it->job, &out);
				image_free(&it->img);
				it->img = out;
			} else {
				nlm_job_discard(&it->job);
				failed_dn++;
			}

			ts = omp_get_wtime() - ts;
			it->time_s[STAGE_DENOISE] += ts;
			wall_dn += ts;
			it->err[STAGE_DENOISE] = r;
			progress_add(progress, it->units);
		}

		/* edge detection, on the denoised or decoded pixels */
		if (set->edges.enabled) {
			if (!eligible(set, it, STAGE_EDGES)) {
				it->err[STAGE_EDGES] = IO_NOT_DONE;
				progress_add(progress, 1);
				continue;
			}

			it->bytes[STAGE_EDGES] = image_bytes(&it->img);
			ts = omp_get_wtime();
			r = gpu_edges(&it->img, &set->edges.setup, &set->gpu_time, on_gpu);
			ts = omp_get_wtime() - ts;

			it->time_s[STAGE_EDGES] = ts;
			wall_ed += ts;
			it->err[STAGE_EDGES] = r;
			failed_ed += (r != IMG_OK);
			progress_add(progress, 1);
		}
	}

	if (set->denoise.enabled)
		stage_done(set, STAGE_DENOISE, wall_dn);
	if (set->edges.enabled)
		stage_done(set, STAGE_EDGES, wall_ed);

	if ((failed_dn || failed_ed) && !set->quiet) {
		progress_clear(progress);
		if (failed_dn)
			report_failures(set, STAGE_DENOISE, first, last);
		if (failed_ed)
			report_failures(set, STAGE_EDGES, first, last);
	}

	return failed_dn + failed_ed;
}

/**
 * @brief Frees the buffers of a range of images.
 *
 * Paths, per-stage results and dimensions are kept for benchmarking.
 *
 * @param[in,out] set   Image set.
 * @param[in]     first First image of the range.
 * @param[in]     last  One past the last image of the range.
 */
void
io_release(ImageSet *set, size_t first, size_t last)
{
	if (last > set->count)
		last = set->count;

	for (size_t i = first; i < last; i++) {
		free(set->items[i].buf);
		set->items[i].buf = NULL;
		set->items[i].buf_len = 0;
		image_free(&set->items[i].img);
		nlm_job_discard(&set->items[i].job);
	}
}

/**
 * @brief Clears all per-run state, keeping paths and the output format.
 *
 * Call before running the stages again on the same set (e.g. repeated
 * benchmark trials). Releases any remaining buffers.
 *
 * @param[in,out] set Image set.
 */
void
io_reset(ImageSet *set)
{
	io_release(set, 0, set->count);

	for (size_t i = 0; i < set->count; i++) {
		ImageItem *it = &set->items[i];

		it->in_format = IMG_FMT_UNKNOWN;
		it->width = it->height = 0;
		it->sigma = 0.0;
		it->detail = NULL;

		for (int s = 0; s < STAGE_COUNT; s++) {
			it->err[s] = IO_NOT_DONE;
			it->sys_errno[s] = 0;
			it->time_s[s] = 0.0;
			it->bytes[s] = 0;
		}
	}

	for (int s = 0; s < STAGE_COUNT; s++) {
		set->performed[s] = 0;
		set->wall_time_s[s] = 0.0;
	}

	memset(&set->gpu_time, 0, sizeof(set->gpu_time));
}

/**
 * @brief Progress units of one image over the stages that will run.
 *
 * Every I/O and codec stage, and edge detection, counts 1 unit per image,
 * and denoising counts nlm_units() (the number of search offsets, 440 with
 * the defaults), which reflects that it dominates the run time.
 *
 * @param[in] set   Image set (its denoise configuration is used).
 * @param[in] write Non-zero if the encode and write stages will run.
 *
 * @return Units per image.
 */
size_t
io_progress_units(const ImageSet *set, int write)
{
	size_t u = 2 + (write ? 2 : 0);

	if (set->denoise.enabled)
		u += nlm_units(&set->denoise.params);
	if (set->edges.enabled)
		u += 1;

	return u;
}

/**
 * @brief Returns the short name of a stage ("read", "decode", ...).
 *
 * @param[in] stage STAGE_* value.
 *
 * @return Static name string.
 */
const char*
io_stage_name(int stage)
{
	return (stage >= 0 && stage < STAGE_COUNT) ? stage_names[stage] : "unknown";
}

/**
 * @brief Frees all paths and buffers of the set.
 *
 * Safe to call on a zeroed set.
 *
 * @param[in,out] set Image set.
 */
void
io_free(ImageSet *set)
{
	if (!set || !set->items)
		return;

	for (size_t i = 0; i < set->count; i++) {
		free(set->items[i].in_path);
		free(set->items[i].out_path);
		free(set->items[i].buf);
		image_free(&set->items[i].img);
	}

	free(set->items);
	set->items = NULL;
	set->count = 0;
}
