/**
 * @file stb.c
 * @brief Compilation unit for the vendored stb single-header libraries.
 *
 * stb_image v2.30 and stb_image_write v1.16 by Sean Barrett
 * (https://github.com/nothings/stb), public domain / MIT.
 *
 * Only the formats the program supports are compiled in. All I/O goes
 * through memory buffers, so stdio support is disabled. This file is built
 * with warnings off (see the Makefile), as it is third-party code.
 */

#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_ONLY_TGA
#define STBI_NO_STDIO
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STBI_WRITE_NO_STDIO
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
