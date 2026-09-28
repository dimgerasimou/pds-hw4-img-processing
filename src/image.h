/**
 * @file image.h
 * @brief 8-bit grayscale images and in-memory codecs.
 *
 * Decoding and encoding work on memory buffers only, so that file I/O and
 * (de)compression are timed as separate stages (see io.h). The format is
 * detected from the contents, not the extension:
 *
 *   PGM (P5/P2, 8/16-bit)   decode + encode   own parser
 *   PNG (8/16-bit)          decode + encode   stb
 *   JPEG, BMP, TGA          decode            stb
 *
 * Color is converted to grayscale and 16-bit samples to 8 bits.
 */

#ifndef IMAGE_H
#define IMAGE_H

#include <stddef.h>

typedef struct {
	unsigned int width;
	unsigned int height;
	unsigned char *data;
} Image;

enum {
	IMG_FMT_UNKNOWN = 0,
	IMG_FMT_PGM,
	IMG_FMT_PNG,
	IMG_FMT_JPEG,
	IMG_FMT_BMP,
	IMG_FMT_TGA,
	IMG_FMT_COUNT
};

enum {
	IMG_OK = 0,
	IMG_ERR_SYS,         /* see the reported errno */
	IMG_ERR_FORMAT,
	IMG_ERR_MAXVAL,
	IMG_ERR_SIZE,
	IMG_ERR_TRUNC,
	IMG_ERR_NOMEM,
	IMG_ERR_UNSUPPORTED,
	IMG_ERR_GPU
};

/**
 * @brief Decodes an image from memory.
 *
 * On failure @p img is zeroed and @p detail (may be NULL) points to the
 * decoder's message, if any. Thread-safe.
 */
int image_decode(const unsigned char *buf, size_t len, Image *img,
                 int *fmt, const char **detail);

/** @brief Encodes to PGM or PNG into a new buffer (caller frees). Thread-safe. */
int image_encode(const Image *img, int fmt, unsigned char **out, size_t *len);

/** @brief Frees the pixels and zeroes the image. Safe on a zeroed image. */
void image_free(Image *img);

size_t image_bytes(const Image *img);

const char* image_format_name(int fmt);
const char* image_format_ext(int fmt);
int image_format_from_name(const char *name);

/** @brief Format by file extension; used only to select files and for -o names. */
int image_format_from_ext(const char *path);

int image_format_writable(int fmt);

/** @brief Error message. Not thread-safe for IMG_ERR_SYS (strerror()). */
const char* image_strerror(int err, int sys_errno);

#endif /* IMAGE_H */
