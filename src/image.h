/**
 * @file image.h
 * @brief Grayscale image container and in-memory image codecs.
 *
 * Images are stored internally as 8-bit single-channel buffers in row-major
 * order: the pixel at (x, y) is data[y * width + x]. This is the only
 * representation the rest of the program works with.
 *
 * Decoding and encoding work on memory buffers only; reading and writing
 * files is done separately (see io.h), so that file I/O and (de)compression
 * can be timed as distinct stages.
 *
 * Supported formats, detected from the file contents (not the extension):
 *   - PGM  (Netpbm P5 binary 8/16-bit, P2 ASCII)  decode + encode, own parser
 *   - PNG  (8/16-bit, any color type)             decode + encode, stb_image
 *   - JPEG (baseline and progressive)             decode only,     stb_image
 *   - BMP, TGA                                    decode only,     stb_image
 *
 * Color images are converted to grayscale and 16-bit samples are rescaled
 * to 8 bits on decode.
 */

#ifndef IMAGE_H
#define IMAGE_H

#include <stddef.h>

/* ------------------------------------------------------------------------- */
/*                              Data Structures                              */
/* ------------------------------------------------------------------------- */

/**
 * @struct Image
 * @brief 8-bit grayscale image.
 *
 * The pixel buffer is owned by the structure and released by image_free().
 */
typedef struct {
	unsigned int width;   /**< Width in pixels */
	unsigned int height;  /**< Height in pixels */
	unsigned char *data;  /**< Row-major pixel buffer (width * height bytes) */
} Image;

/**
 * @enum Image formats
 * @brief File formats known to the codec layer.
 */
enum {
	IMG_FMT_UNKNOWN = 0, /**< Unknown / not specified */
	IMG_FMT_PGM,         /**< Netpbm PGM */
	IMG_FMT_PNG,         /**< PNG */
	IMG_FMT_JPEG,        /**< JPEG */
	IMG_FMT_BMP,         /**< BMP */
	IMG_FMT_TGA,         /**< TGA */
	IMG_FMT_COUNT        /**< Sentinel: number of formats */
};

/**
 * @enum Image error codes
 * @brief Result codes returned by the codec functions.
 */
enum {
	IMG_OK = 0,          /**< Success */
	IMG_ERR_SYS,         /**< System error, see the reported errno */
	IMG_ERR_FORMAT,      /**< Unrecognized format or malformed data */
	IMG_ERR_MAXVAL,      /**< Unsupported PGM maxval (must be 1-65535) */
	IMG_ERR_SIZE,        /**< Invalid or unsupported image dimensions */
	IMG_ERR_TRUNC,       /**< Unexpected end of data */
	IMG_ERR_NOMEM,       /**< Memory allocation failed */
	IMG_ERR_UNSUPPORTED  /**< Format cannot be written */
};

/* ------------------------------------------------------------------------- */
/*                            Public API Functions                           */
/* ------------------------------------------------------------------------- */

/**
 * @brief Decodes an image from a memory buffer.
 *
 * The format is detected from the data itself. On success the caller owns
 * img->data and must release it with image_free(). On failure @p img is left
 * zeroed.
 *
 * @note Thread-safe: may be called concurrently on different buffers.
 *
 * @param[in]  buf    Encoded image data.
 * @param[in]  len    Size of @p buf in bytes.
 * @param[out] img    Decoded image.
 * @param[out] fmt    Detected format (may be NULL).
 * @param[out] detail Static string with decoder details on failure, or NULL
 *                    (may be NULL).
 *
 * @return IMG_OK on success, otherwise one of the IMG_ERR_* codes.
 */
int image_decode(const unsigned char *buf, size_t len, Image *img,
                 int *fmt, const char **detail);

/**
 * @brief Encodes an image into a newly allocated memory buffer.
 *
 * @note Thread-safe: may be called concurrently on different images.
 *
 * @param[in]  img Image to encode.
 * @param[in]  fmt Output format (IMG_FMT_PGM or IMG_FMT_PNG).
 * @param[out] out Newly allocated encoded data (caller frees with free()).
 * @param[out] len Size of @p out in bytes.
 *
 * @return IMG_OK on success, otherwise one of the IMG_ERR_* codes.
 */
int image_encode(const Image *img, int fmt, unsigned char **out, size_t *len);

/**
 * @brief Releases the pixel buffer and zeroes the structure.
 *
 * Safe to call on a zeroed or already freed image.
 *
 * @param[in,out] img Image to free.
 */
void image_free(Image *img);

/**
 * @brief Returns the size of the pixel buffer in bytes.
 *
 * @param[in] img Image.
 *
 * @return width * height.
 */
size_t image_bytes(const Image *img);

/**
 * @brief Returns the short name of a format ("pgm", "png", ...).
 *
 * @param[in] fmt IMG_FMT_* value.
 *
 * @return Static name string ("unknown" for invalid values).
 */
const char* image_format_name(int fmt);

/**
 * @brief Returns the file extension of a format, including the dot.
 *
 * @param[in] fmt IMG_FMT_* value.
 *
 * @return Static extension string ("" for invalid values).
 */
const char* image_format_ext(int fmt);

/**
 * @brief Parses a format name (case-insensitive).
 *
 * @param[in] name Format name, e.g. "png".
 *
 * @return IMG_FMT_* value, or IMG_FMT_UNKNOWN.
 */
int image_format_from_name(const char *name);

/**
 * @brief Guesses a format from a file name's extension.
 *
 * Used only to select candidate files and to interpret -o file names; the
 * actual input format is always detected from the contents.
 *
 * @param[in] path File name or path.
 *
 * @return IMG_FMT_* value, or IMG_FMT_UNKNOWN.
 */
int image_format_from_ext(const char *path);

/**
 * @brief Checks whether a format can be encoded.
 *
 * @param[in] fmt IMG_FMT_* value.
 *
 * @return 1 if writable, else 0.
 */
int image_format_writable(int fmt);

/**
 * @brief Returns a human-readable description of an image error code.
 *
 * @note Not thread-safe for IMG_ERR_SYS (uses strerror()). Call from a
 *       single thread, e.g. when reporting errors after a parallel region.
 *
 * @param[in] err       IMG_* error code.
 * @param[in] sys_errno errno value, used only for IMG_ERR_SYS.
 *
 * @return Static description string.
 */
const char* image_strerror(int err, int sys_errno);

#endif /* IMAGE_H */
