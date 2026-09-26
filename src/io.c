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
#include "io.h"
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
	"read", "decode", "encode", "write"
};

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

	it->bytes[STAGE_DECODE] = image_bytes(&it->img);
	return r;
}

/**
 * @brief Encode stage for one image: img -> buf.
 */
static int
do_encode(ImageItem *it, int fmt)
{
	it->bytes[STAGE_ENCODE] = image_bytes(&it->img);
	return image_encode(&it->img, fmt, &it->buf, &it->buf_len);
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
 * @brief Checks whether an image should take part in a stage.
 */
static int
eligible(const ImageItem *it, int stage)
{
	switch (stage) {
	case STAGE_READ:   return 1;
	case STAGE_DECODE: return it->err[STAGE_READ] == IMG_OK;
	case STAGE_ENCODE: return it->out_path && it->err[STAGE_DECODE] == IMG_OK;
	case STAGE_WRITE:  return it->err[STAGE_ENCODE] == IMG_OK;
	default:           return 0;
	}
}

/**
 * @brief Reports the failures of a stage, in input order.
 */
static void
report_failures(const ImageSet *set, int stage)
{
	for (size_t i = 0; i < set->count; i++) {
		const ImageItem *it = &set->items[i];
		int e = it->err[stage];

		if (e == IMG_OK || e == IO_NOT_DONE)
			continue;

		switch (stage) {
		case STAGE_READ:
			uerrf("cannot read \"%s\": %s", it->in_path,
			      image_strerror(e, it->sys_errno[stage]));
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

/**
 * @brief Runs one stage over every eligible image in parallel.
 *
 * An image is eligible if it passed the previous stage (and, for encode and
 * write, has an output path). Failures are reported after the parallel
 * region, in input order. The wall time of the stage is stored in the set.
 *
 * Data volume recorded per image: bytes read (read), pixel bytes produced
 * (decode), pixel bytes consumed (encode), bytes written (write).
 *
 * @param[in,out] set      Image set.
 * @param[in]     stage    STAGE_* value.
 * @param[in]     progress Non-zero to show a progress bar.
 *
 * @return Number of images that failed this stage.
 */
size_t
io_run(ImageSet *set, int stage, int progress)
{
	Progress p;
	size_t failed = 0, todo = 0;
	double t0;

	if (stage < 0 || stage >= STAGE_COUNT) {
		DERRF("invalid stage: %d", stage);
		return 0;
	}

	for (size_t i = 0; i < set->count; i++)
		if (eligible(&set->items[i], stage))
			todo++;

	progress_init(&p, stage_names[stage], todo, progress);
	t0 = omp_get_wtime();

	#pragma omp parallel for schedule(dynamic) reduction(+:failed)
	for (size_t i = 0; i < set->count; i++) {
		ImageItem *it = &set->items[i];
		double ts;
		int r;

		if (!eligible(it, stage)) {
			it->err[stage] = IO_NOT_DONE;
			continue;
		}

		ts = omp_get_wtime();
		switch (stage) {
		case STAGE_READ:   r = do_read(it);                     break;
		case STAGE_DECODE: r = do_decode(it);                   break;
		case STAGE_ENCODE: r = do_encode(it, set->out_format);  break;
		default:           r = do_write(it);                    break;
		}
		it->time_s[stage] = omp_get_wtime() - ts;
		it->err[stage] = r;

		if (r != IMG_OK)
			failed++;

		progress_tick(&p);
	}

	set->wall_time_s[stage] = omp_get_wtime() - t0;
	set->performed[stage] = 1;

	progress_finish(&p);
	report_failures(set, stage);

	return failed;
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
