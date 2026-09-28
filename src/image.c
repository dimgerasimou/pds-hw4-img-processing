/**
 * @file image.c
 * @brief Image codecs: PGM here, everything else via stb.
 *
 * stb_image keeps its error state thread-local, so concurrent decoding is safe.
 */

#define _POSIX_C_SOURCE 200809L

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "external/stb_image.h"
#include "external/stb_image_write.h"
#include "image.h"

/* guards against absurd headers */
#define IMG_MAX_DIM 65536U

static const char *format_names[IMG_FMT_COUNT] = {
	"unknown", "pgm", "png", "jpeg", "bmp", "tga"
};

static const char *format_exts[IMG_FMT_COUNT] = {
	"", ".pgm", ".png", ".jpg", ".bmp", ".tga"
};

typedef struct {
	const unsigned char *p;
	const unsigned char *end;
} Cursor;

typedef struct {
	unsigned char *data;
	size_t len;
	size_t cap;
	int failed;
} Sink;

static int
is_space(int c)
{
	return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

static int
is_digit(int c)
{
	return c >= '0' && c <= '9';
}

static int
cur_getc(Cursor *c)
{
	return (c->p < c->end) ? *c->p++ : EOF;
}

static int
skip_space(Cursor *c)
{
	int ch;

	for (;;) {
		ch = cur_getc(c);

		if (ch == '#') {
			while ((ch = cur_getc(c)) != EOF && ch != '\n')
				;
			if (ch == EOF)
				return EOF;
			continue;
		}

		if (!is_space(ch))
			return ch;
	}
}

/*
 * Consumes the character ending the number: after the last header field
 * that is the single whitespace byte before the binary payload. A '#' is
 * pushed back, so that the comment is skipped as usual.
 */
static int
read_uint(Cursor *c, unsigned int *out, int *delim)
{
	unsigned long v = 0;
	int ch;

	ch = skip_space(c);
	if (ch == EOF)
		return IMG_ERR_TRUNC;
	if (!is_digit(ch))
		return IMG_ERR_FORMAT;

	do {
		v = v * 10 + (unsigned long)(ch - '0');
		if (v > UINT_MAX)
			return IMG_ERR_FORMAT;
		ch = cur_getc(c);
	} while (is_digit(ch));

	if (ch == '#')
		c->p--;

	*out = (unsigned int)v;
	*delim = ch;
	return IMG_OK;
}

static unsigned char
rescale(unsigned int v, unsigned int maxval)
{
	if (v >= maxval)
		return 255;
	return (unsigned char)((v * 255U + maxval / 2U) / maxval);
}

/* TGA has no signature: unknown data is left to stb's heuristics. */
static int
detect_format(const unsigned char *buf, size_t len)
{
	static const unsigned char png_sig[8] = { 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n' };

	if (len >= 2 && buf[0] == 'P' && (buf[1] == '5' || buf[1] == '2'))
		return IMG_FMT_PGM;
	if (len >= 8 && memcmp(buf, png_sig, 8) == 0)
		return IMG_FMT_PNG;
	if (len >= 3 && buf[0] == 0xff && buf[1] == 0xd8 && buf[2] == 0xff)
		return IMG_FMT_JPEG;
	if (len >= 2 && buf[0] == 'B' && buf[1] == 'M')
		return IMG_FMT_BMP;
	return IMG_FMT_UNKNOWN;
}

static int
decode_pgm(const unsigned char *buf, size_t len, Image *img)
{
	Cursor c = { buf + 2, buf + len };
	unsigned int width, height, maxval;
	int binary = (buf[1] == '5');
	int delim, ret;
	size_t n;

	if ((ret = read_uint(&c, &width, &delim)) != IMG_OK)
		return ret;
	if ((ret = read_uint(&c, &height, &delim)) != IMG_OK)
		return ret;
	if ((ret = read_uint(&c, &maxval, &delim)) != IMG_OK)
		return ret;

	if (!is_space(delim))
		return (delim == EOF) ? IMG_ERR_TRUNC : IMG_ERR_FORMAT;

	if (maxval == 0 || maxval > 65535)
		return IMG_ERR_MAXVAL;

	if (width == 0 || height == 0 || width > IMG_MAX_DIM || height > IMG_MAX_DIM)
		return IMG_ERR_SIZE;

	n = (size_t)width * height;
	img->data = malloc(n);
	if (!img->data)
		return IMG_ERR_NOMEM;
	img->width = width;
	img->height = height;

	if (!binary) {
		for (size_t i = 0; i < n; i++) {
			unsigned int v;
			if ((ret = read_uint(&c, &v, &delim)) != IMG_OK)
				return ret;
			img->data[i] = rescale(v, maxval);
		}
		return IMG_OK;
	}

	if (maxval <= 255) {
		if ((size_t)(c.end - c.p) < n)
			return IMG_ERR_TRUNC;

		if (maxval == 255)
			memcpy(img->data, c.p, n);
		else
			for (size_t i = 0; i < n; i++)
				img->data[i] = rescale(c.p[i], maxval);
		return IMG_OK;
	}

	if ((size_t)(c.end - c.p) / 2 < n)
		return IMG_ERR_TRUNC;

	for (size_t i = 0; i < n; i++) {
		unsigned int v = ((unsigned int)c.p[2 * i] << 8) | c.p[2 * i + 1];
		img->data[i] = rescale(v, maxval);
	}

	return IMG_OK;
}

static int
decode_stb(const unsigned char *buf, size_t len, Image *img, const char **detail)
{
	int w, h, comp;

	if (len > INT_MAX)
		return IMG_ERR_SIZE;

	if (stbi_is_16_bit_from_memory(buf, (int)len)) {
		stbi_us *d16 = stbi_load_16_from_memory(buf, (int)len, &w, &h, &comp, 1);
		size_t n;

		if (!d16)
			goto fail;

		n = (size_t)w * (size_t)h;
		img->data = malloc(n);
		if (!img->data) {
			stbi_image_free(d16);
			return IMG_ERR_NOMEM;
		}

		for (size_t i = 0; i < n; i++)
			img->data[i] = rescale(d16[i], 65535);

		stbi_image_free(d16);
	} else {
		img->data = stbi_load_from_memory(buf, (int)len, &w, &h, &comp, 1);
		if (!img->data)
			goto fail;
	}

	img->width = (unsigned int)w;
	img->height = (unsigned int)h;
	return IMG_OK;

fail:
	if (detail)
		*detail = stbi_failure_reason();
	return IMG_ERR_FORMAT;
}

static int
encode_pgm(const Image *img, unsigned char **out, size_t *len)
{
	char hdr[48];
	size_t n = image_bytes(img);
	int hlen;

	hlen = snprintf(hdr, sizeof(hdr), "P5\n%u %u\n255\n", img->width, img->height);
	if (hlen < 0 || (size_t)hlen >= sizeof(hdr))
		return IMG_ERR_SIZE;

	*out = malloc((size_t)hlen + n);
	if (!*out)
		return IMG_ERR_NOMEM;

	memcpy(*out, hdr, (size_t)hlen);
	memcpy(*out + hlen, img->data, n);
	*len = (size_t)hlen + n;
	return IMG_OK;
}

static void
sink_write(void *ctx, void *data, int size)
{
	Sink *s = ctx;
	size_t need;

	if (s->failed || size <= 0)
		return;

	need = s->len + (size_t)size;
	if (need > s->cap) {
		size_t ncap = s->cap ? s->cap : 4096;
		unsigned char *nd;

		while (ncap < need)
			ncap *= 2;

		nd = realloc(s->data, ncap);
		if (!nd) {
			s->failed = 1;
			return;
		}
		s->data = nd;
		s->cap = ncap;
	}

	memcpy(s->data + s->len, data, (size_t)size);
	s->len = need;
}

static int
encode_png(const Image *img, unsigned char **out, size_t *len)
{
	Sink s = { NULL, 0, 0, 0 };

	if (img->width > INT_MAX || img->height > INT_MAX)
		return IMG_ERR_SIZE;

	if (!stbi_write_png_to_func(sink_write, &s, (int)img->width, (int)img->height,
	                            1, img->data, (int)img->width) || s.failed) {
		free(s.data);
		return IMG_ERR_NOMEM;
	}

	*out = s.data;
	*len = s.len;
	return IMG_OK;
}

int
image_decode(const unsigned char *buf, size_t len, Image *img,
             int *fmt, const char **detail)
{
	int f, ret;

	img->width = img->height = 0;
	img->data = NULL;

	if (detail)
		*detail = NULL;

	if (!buf || len == 0)
		return IMG_ERR_TRUNC;

	f = detect_format(buf, len);
	ret = (f == IMG_FMT_PGM) ? decode_pgm(buf, len, img)
	                         : decode_stb(buf, len, img, detail);

	if (ret == IMG_OK && f == IMG_FMT_UNKNOWN)
		f = IMG_FMT_TGA;

	if (fmt)
		*fmt = f;

	if (ret != IMG_OK)
		image_free(img);

	return ret;
}

int
image_encode(const Image *img, int fmt, unsigned char **out, size_t *len)
{
	*out = NULL;
	*len = 0;

	if (!img || !img->data || !img->width || !img->height)
		return IMG_ERR_SIZE;

	switch (fmt) {
	case IMG_FMT_PGM: return encode_pgm(img, out, len);
	case IMG_FMT_PNG: return encode_png(img, out, len);
	default:          return IMG_ERR_UNSUPPORTED;
	}
}

void
image_free(Image *img)
{
	if (!img)
		return;

	free(img->data);
	img->data = NULL;
	img->width = img->height = 0;
}

size_t
image_bytes(const Image *img)
{
	return (size_t)img->width * img->height;
}

const char*
image_format_name(int fmt)
{
	return (fmt > 0 && fmt < IMG_FMT_COUNT) ? format_names[fmt] : format_names[0];
}

const char*
image_format_ext(int fmt)
{
	return (fmt > 0 && fmt < IMG_FMT_COUNT) ? format_exts[fmt] : format_exts[0];
}

int
image_format_from_name(const char *name)
{
	if (!name)
		return IMG_FMT_UNKNOWN;

	for (int f = 1; f < IMG_FMT_COUNT; f++)
		if (strcasecmp(name, format_names[f]) == 0)
			return f;

	if (strcasecmp(name, "jpg") == 0)
		return IMG_FMT_JPEG;

	return IMG_FMT_UNKNOWN;
}

int
image_format_from_ext(const char *path)
{
	const char *slash = strrchr(path, '/');
	const char *base = slash ? slash + 1 : path;
	const char *dot = strrchr(base, '.');

	if (!dot || dot == base)
		return IMG_FMT_UNKNOWN;

	dot++;

	if (strcasecmp(dot, "pgm") == 0 || strcasecmp(dot, "pnm") == 0)
		return IMG_FMT_PGM;
	if (strcasecmp(dot, "png") == 0)
		return IMG_FMT_PNG;
	if (strcasecmp(dot, "jpg") == 0 || strcasecmp(dot, "jpeg") == 0)
		return IMG_FMT_JPEG;
	if (strcasecmp(dot, "bmp") == 0)
		return IMG_FMT_BMP;
	if (strcasecmp(dot, "tga") == 0)
		return IMG_FMT_TGA;

	return IMG_FMT_UNKNOWN;
}

int
image_format_writable(int fmt)
{
	return fmt == IMG_FMT_PGM || fmt == IMG_FMT_PNG;
}

const char*
image_strerror(int err, int sys_errno)
{
	switch (err) {
	case IMG_OK:              return "success";
	case IMG_ERR_SYS:         return strerror(sys_errno);
	case IMG_ERR_FORMAT:      return "unrecognized or corrupt image data";
	case IMG_ERR_MAXVAL:      return "unsupported PGM maxval";
	case IMG_ERR_SIZE:        return "invalid image dimensions";
	case IMG_ERR_TRUNC:       return "unexpected end of data";
	case IMG_ERR_NOMEM:       return "out of memory";
	case IMG_ERR_UNSUPPORTED: return "not supported";
	case IMG_ERR_GPU:         return "GPU error";
	default:                  return "unknown error";
	}
}
