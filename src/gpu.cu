/**
 * @file gpu.cu
 * @brief CUDA implementation of NLM denoising and Canny edge detection.
 *
 * This is the only CUDA translation unit; everything else is C. It contains
 * the kernel and thin wrappers with C linkage (see gpu.h).
 *
 * Kernel design: blocks of BLOCK_W x BLOCK_H threads, each thread computing
 * PIX_Y vertically adjacent output pixels. A block loads its tile of the
 * padded image, plus a border of patch + search radius, into shared memory
 * once. It then processes the offsets of the search window GROUP at a
 * time: all threads compute the horizontal patch sums of the group's
 * offsets straight from the image tile (a sliding window over squared
 * differences held in registers), and after a barrier every thread sums
 * its columns over the patch height, sliding down from one of its pixels
 * to the next. That is each pixel's exact integer patch sum, the same value
 * the CPU obtains from integral images. Weights are read from the table
 * built by the CPU, and every thread keeps its accumulators in registers
 * for the whole loop, so they cost no memory traffic.
 *
 * The kernel is a template over the patch radius (instantiated for every
 * radius the program accepts), so that the sliding window is a register
 * array and every division in the loops is by a constant.
 *
 * Offsets are visited in the same order as on the CPU and the arithmetic is
 * the same, and the file is compiled without fused multiply-add
 * (-fmad=false), so the output is bit-for-bit identical to the CPU's.
 *
 * The phases of the kernel are __host__ __device__ functions of a tile and a
 * thread position, so that they can also be executed on the host (one
 * thread after the other, phase by phase) to test the kernel's indexing
 * without a GPU.
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* project headers are C: give their declarations C linkage */
extern "C" {
#include "canny.h"
#include "error.h"
#include "nlm.h"
}
#include "gpu.h"

/*
 * Block shape: BLOCK_W x BLOCK_H threads, each computing PIX_Y vertically
 * adjacent output pixels, so a block covers a tile of TILE_W x TILE_H
 * pixels. Taller tiles make the patch border (2P rows of extra horizontal
 * sums) a smaller share of the work.
 */
#define BLOCK_W 32
#define BLOCK_H 16
#define PIX_Y   2
#define TILE_W  BLOCK_W
#define TILE_H  (BLOCK_H * PIX_Y)

/*
 * GROUP offsets are processed per barrier round; the number of offsets,
 * 4 S (S + 1), is a multiple of 8 for every S, so GROUP = 4 divides it.
 * SEG horizontal sums are computed per work item, sliding a window over
 * SEG + 2P squared differences held in registers.
 */
#define GROUP 4
#define SEG   4

/* Largest patch radius with a kernel instantiation (the program's limit) */
#define MAX_P 10

/**
 * @struct Tile
 * @brief Shared-memory layout of one block.
 */
struct Tile {
	unsigned char *img; /**< Padded image tile: sh x sw bytes */
	int *hsum;          /**< Horizontal patch sums: GROUP x dh x TILE_W */
	int sw, sh;         /**< Image tile size (tile + 2 (P + S)) */
	int s;              /**< Search radius */
};

/* ------------------------------------------------------------------------- */
/*                               Kernel Phases                               */
/* ------------------------------------------------------------------------- */

/*
 * The phases are templates over the patch radius P, so that the sliding
 * window lives in registers and every division is by a constant. They take
 * the thread's index in the block (or position) and can also be executed on
 * the host, one thread after the other, to test the indexing without a GPU.
 */

/**
 * @brief Bytes of shared memory needed by one block.
 */
__host__ __device__ static inline size_t
tile_bytes(int p, int s)
{
	int r = p + s;
	size_t img = (size_t)(TILE_W + 2 * r) * (TILE_H + 2 * r);
	size_t hs = (size_t)GROUP * TILE_W * (TILE_H + 2 * p);

	img = (img + 3) & ~(size_t)3; /* align the int array */
	return img + hs * sizeof(int);
}

/**
 * @brief Lays out a tile in @p smem.
 */
__host__ __device__ static inline Tile
tile_make(unsigned char *smem, int p, int s)
{
	Tile t;
	int r = p + s;
	size_t img;

	t.s = s;
	t.sw = TILE_W + 2 * r;
	t.sh = TILE_H + 2 * r;

	img = ((size_t)t.sw * t.sh + 3) & ~(size_t)3;
	t.img = smem;
	t.hsum = (int *)(smem + img);
	return t;
}

/**
 * @brief Phase 1: loads the block's padded image tile. Outside the padded
 *        image (right/bottom edge blocks), zeros.
 */
__host__ __device__ static inline void
tile_load(const Tile *t, const unsigned char *pad, int pw, int ph,
          int x0, int y0, int tx, int ty)
{
	for (int r = ty; r < t->sh; r += BLOCK_H) {
		const int gr = y0 + r;

		for (int c = tx; c < t->sw; c += BLOCK_W) {
			const int gc = x0 + c;
			t->img[r * t->sw + c] = (gr < ph && gc < pw) ? pad[(size_t)gr * pw + gc] : 0;
		}
	}
}

/**
 * @brief The next GROUP offsets in raster order, skipping the center.
 *
 * (*cx, *cy) is the running position in the search window.
 */
__host__ __device__ static inline void
next_offsets(int s, int sw, int *cx, int *cy, int dx[GROUP], int dy[GROUP], int shift[GROUP])
{
	for (int k = 0; k < GROUP; k++) {
		if (*cx == 0 && *cy == 0) {
			if (++*cx > s) { *cx = -s; ++*cy; }
		}
		dx[k] = *cx;
		dy[k] = *cy;
		shift[k] = *cy * sw + *cx;
		if (++*cx > s) { *cx = -s; ++*cy; }
	}
}

/**
 * @brief Phase 2: horizontal patch sums of GROUP offsets, straight from the
 *        image tile.
 *
 * A work item is (offset k, row r, SEG consecutive columns). Its SEG + 2P
 * squared differences are computed once into registers, and the SEG sums
 * of 2P + 1 of them are obtained by sliding a window. Row r corresponds to
 * image tile row r + S, column x to image tile column x + S; the sum for
 * (k, r, x) covers columns x .. x + 2P.
 */
template <int P>
__host__ __device__ static inline void
tile_hsum(const Tile *t, const int shift[GROUP], int tid, int nt)
{
	constexpr int side = 2 * P + 1;
	constexpr int dh = TILE_H + 2 * P;
	constexpr int segs = TILE_W / SEG;
	constexpr int items = dh * segs;

	for (int item = tid; item < GROUP * items; item += nt) {
		const int k = item / items;
		const int rem = item - k * items;
		const int r = rem / segs;
		const int x0 = (rem - r * segs) * SEG;
		const unsigned char *a = t->img + (r + t->s) * t->sw + (x0 + t->s);
		const unsigned char *b = a + shift[k];
		int *h = t->hsum + (k * dh + r) * TILE_W + x0;
		int d[SEG + 2 * P];
		int sum = 0;

		#pragma unroll
		for (int m = 0; m < SEG + 2 * P; m++) {
			const int v = (int)a[m] - (int)b[m];
			d[m] = v * v;
		}

		#pragma unroll
		for (int j = 0; j < side; j++)
			sum += d[j];
		h[0] = sum;

		#pragma unroll
		for (int i = 1; i < SEG; i++) {
			sum += d[i + side - 1] - d[i - 1];
			h[i] = sum;
		}
	}
}

/**
 * @brief Phase 3 (per thread): the patch sums of the thread's PIX_Y pixels
 *        for offset k of the group, rows ty * PIX_Y ... of column tx, from
 *        2P+1 vertically adjacent horizontal sums. The window slides down
 *        one row per pixel.
 */
template <int P>
__host__ __device__ static inline void
tile_patch(const Tile *t, int k, int tx, int ty, int ssd[PIX_Y])
{
	constexpr int side = 2 * P + 1;
	constexpr int dh = TILE_H + 2 * P;
	const int *h = t->hsum + (k * dh + ty * PIX_Y) * TILE_W + tx;
	int sum = 0;

	#pragma unroll
	for (int i = 0; i < side; i++)
		sum += h[i * TILE_W];
	ssd[0] = sum;

	#pragma unroll
	for (int j = 1; j < PIX_Y; j++) {
		sum += h[(j - 1 + side) * TILE_W] - h[(j - 1) * TILE_W];
		ssd[j] = sum;
	}
}

/**
 * @brief Adds a candidate to a pixel's accumulators (same arithmetic and
 *        order as the CPU).
 */
__host__ __device__ static inline void
accumulate(float w, unsigned char q, float *wsum, float *vsum, float *wmax)
{
	*wsum += w;
	*vsum += w * (float)q;
	if (w > *wmax)
		*wmax = w;
}

/**
 * @brief Final value of a pixel (same as the CPU): the center counts as
 *        much as its most similar neighbor.
 */
__host__ __device__ static inline unsigned char
finish(float wsum, float vsum, float wmax, unsigned char center)
{
	long value;

	if (wmax == 0.0f)
		wmax = 1.0f;
	wsum += wmax;
	vsum += wmax * (float)center;

	value = lroundf(vsum / wsum);
	return (unsigned char)(value < 0 ? 0 : value > 255 ? 255 : value);
}

/* ------------------------------------------------------------------------- */
/*                                  Kernel                                   */
/* ------------------------------------------------------------------------- */

/**
 * @brief NLM kernel: every thread computes PIX_Y pixels over all offsets.
 *
 * Offsets are processed GROUP at a time, in raster order skipping the
 * center (the same order as on the CPU): all threads compute the group's
 * horizontal sums, then every thread accumulates its pixels over the
 * group's offsets in order. Two barriers per group.
 *
 * @param pad    Mirrored, padded image ((H + 2R) x pw).
 * @param pw     Padded width.
 * @param ph     Padded height.
 * @param w      Image width.
 * @param h      Image height.
 * @param s      Search radius.
 * @param cutoff Largest patch sum with a non-negligible weight.
 * @param wtab   Weight per patch sum 0..cutoff.
 * @param out    Output image (w x h).
 */
template <int P>
__global__ void
nlm_kernel(const unsigned char *__restrict__ pad, int pw, int ph, int w, int h,
           int s, int cutoff, const float *__restrict__ wtab,
           unsigned char *__restrict__ out)
{
	extern __shared__ __align__(4) unsigned char smem[];
	const Tile t = tile_make(smem, P, s);
	const int tx = threadIdx.x, ty = threadIdx.y;
	const int tid = ty * BLOCK_W + tx, nt = BLOCK_W * BLOCK_H;
	const int x0 = blockIdx.x * TILE_W, y0 = blockIdx.y * TILE_H;
	const int r = P + s;
	const int noff = 4 * s * (s + 1);
	float wsum[PIX_Y], vsum[PIX_Y], wmax[PIX_Y];
	int cx = -s, cy = -s;

	for (int k = 0; k < PIX_Y; k++)
		wsum[k] = vsum[k] = wmax[k] = 0.0f;

	tile_load(&t, pad, pw, ph, x0, y0, tx, ty);
	__syncthreads();

	for (int g = 0; g < noff; g += GROUP) {
		int dx[GROUP], dy[GROUP], shift[GROUP];

		next_offsets(s, t.sw, &cx, &cy, dx, dy, shift);

		tile_hsum<P>(&t, shift, tid, nt);
		__syncthreads();                /* horizontal sums of the group complete */

		for (int k = 0; k < GROUP; k++) {
			int ssd[PIX_Y];

			tile_patch<P>(&t, k, tx, ty, ssd);

			for (int j = 0; j < PIX_Y; j++)
				if (ssd[j] <= cutoff)
					accumulate(__ldg(&wtab[ssd[j]]),
					           t.img[(ty * PIX_Y + j + r + dy[k]) * t.sw + (tx + r + dx[k])],
					           &wsum[j], &vsum[j], &wmax[j]);
		}
		__syncthreads();                /* everyone done reading the sums */
	}

	for (int j = 0; j < PIX_Y; j++) {
		const int y = y0 + ty * PIX_Y + j, x = x0 + tx;

		if (x < w && y < h)
			out[(size_t)y * w + x] =
				finish(wsum[j], vsum[j], wmax[j], t.img[(ty * PIX_Y + j + r) * t.sw + (tx + r)]);
	}
}

/**
 * @brief Launches the kernel instantiation for patch radius P.
 */
template <int P>
static cudaError_t
launch(dim3 grid, dim3 block, size_t smem, const unsigned char *pad, int pw, int ph,
       int w, int h, int s, int cutoff, const float *wtab, unsigned char *out)
{
	if (smem > 48 * 1024) {
		cudaError_t err = cudaFuncSetAttribute(nlm_kernel<P>,
		                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
		                                       (int)smem);
		if (err != cudaSuccess)
			return err;
	}

	nlm_kernel<P><<<grid, block, smem>>>(pad, pw, ph, w, h, s, cutoff, wtab, out);
	return cudaGetLastError();
}

/* ------------------------------------------------------------------------- */
/*                              Canny Kernels                                */
/* ------------------------------------------------------------------------- */

/*
 * The per-pixel steps call the same functions as the CPU (canny_core.h).
 * Hysteresis uses tiles of HYST_T x HYST_T pixels, each thread handling
 * HYST_T / HYST_ROWS pixels of a column.
 */
#define CANNY_BW  32
#define CANNY_BH  8
#define HYST_T    32
#define HYST_ROWS 8

/* Gaussian weights of the current image */
__constant__ int c_weights[2 * CANNY_MAX_RADIUS + 1];

/**
 * @brief Horizontal Gaussian pass, one thread per pixel.
 */
__global__ void
canny_blur_h_kernel(const unsigned char *__restrict__ img, int *__restrict__ hb,
                    int w, int h, int r)
{
	const int x = blockIdx.x * CANNY_BW + threadIdx.x;
	const int y = blockIdx.y * CANNY_BH + threadIdx.y;

	if (x < w && y < h)
		hb[(size_t)y * w + x] = canny_blur_h(img, w, x, y, c_weights, r);
}

/**
 * @brief Vertical Gaussian pass, one thread per pixel.
 */
__global__ void
canny_blur_v_kernel(const int *__restrict__ hb, int *__restrict__ b, int w, int h, int r)
{
	const int x = blockIdx.x * CANNY_BW + threadIdx.x;
	const int y = blockIdx.y * CANNY_BH + threadIdx.y;

	if (x < w && y < h)
		b[(size_t)y * w + x] = canny_blur_v(hb, 0, w, h, x, y, c_weights, r);
}

/**
 * @brief Gradients, non-maximum suppression and thresholds, one thread per
 *        pixel: the class map (none / weak / edge).
 */
__global__ void
canny_classify_kernel(const int *__restrict__ b, unsigned char *__restrict__ map,
                      int w, int h, long long tl2, long long th2)
{
	const int x = blockIdx.x * CANNY_BW + threadIdx.x;
	const int y = blockIdx.y * CANNY_BH + threadIdx.y;

	if (x < w && y < h)
		map[(size_t)y * w + x] = canny_classify(b, 0, w, h, x, y, tl2, th2);
}

/**
 * @brief Hysteresis step for tile position (r, c) (1-based inside the halo):
 *        a weak pixel next to an edge becomes an edge.
 *
 * @return 1 if the pixel changed.
 */
__host__ __device__ static inline int
hyst_relax(unsigned char s[HYST_T + 2][HYST_T + 2], int r, int c)
{
	if (s[r][c] != CANNY_WEAK)
		return 0;

	for (int dy = -1; dy <= 1; dy++)
		for (int dx = -1; dx <= 1; dx++)
			if (s[r + dy][c + dx] == CANNY_EDGE) {
				s[r][c] = CANNY_EDGE;
				return 1;
			}

	return 0;
}

/*
 * Which of a tile's border pixels changed, as bits: a change on a side of
 * the tile can affect the neighboring tile on that side, a change in a
 * corner pixel also the diagonal neighbor. Changes inside affect no one.
 */
#define BORDER_TOP    0x01
#define BORDER_BOTTOM 0x02
#define BORDER_LEFT   0x04
#define BORDER_RIGHT  0x08
#define BORDER_TL     0x10
#define BORDER_TR     0x20
#define BORDER_BL     0x40
#define BORDER_BR     0x80

/**
 * @brief Border bits of tile position (r, c) (1-based inside the halo).
 */
__host__ __device__ static inline int
hyst_border_bits(int r, int c)
{
	int bits = 0;

	if (r == 1)      bits |= BORDER_TOP;
	if (r == HYST_T) bits |= BORDER_BOTTOM;
	if (c == 1)      bits |= BORDER_LEFT;
	if (c == HYST_T) bits |= BORDER_RIGHT;
	if (r == 1 && c == 1)           bits |= BORDER_TL;
	if (r == 1 && c == HYST_T)      bits |= BORDER_TR;
	if (r == HYST_T && c == 1)      bits |= BORDER_BL;
	if (r == HYST_T && c == HYST_T) bits |= BORDER_BR;
	return bits;
}

/**
 * @brief Marks the neighbors of tile (bx, by) that its changed border
 *        pixels (@p bits) can affect, for the next launch.
 */
__host__ __device__ static inline void
hyst_mark(unsigned char *next, int tiles_x, int tiles_y, int bx, int by, int bits)
{
	const int up = by > 0, down = by + 1 < tiles_y, left = bx > 0, right = bx + 1 < tiles_x;

	if ((bits & BORDER_TOP) && up)             next[(by - 1) * tiles_x + bx] = 1;
	if ((bits & BORDER_BOTTOM) && down)        next[(by + 1) * tiles_x + bx] = 1;
	if ((bits & BORDER_LEFT) && left)          next[by * tiles_x + bx - 1] = 1;
	if ((bits & BORDER_RIGHT) && right)        next[by * tiles_x + bx + 1] = 1;
	if ((bits & BORDER_TL) && up && left)      next[(by - 1) * tiles_x + bx - 1] = 1;
	if ((bits & BORDER_TR) && up && right)     next[(by - 1) * tiles_x + bx + 1] = 1;
	if ((bits & BORDER_BL) && down && left)    next[(by + 1) * tiles_x + bx - 1] = 1;
	if ((bits & BORDER_BR) && down && right)   next[(by + 1) * tiles_x + bx + 1] = 1;
}

/**
 * @brief Hysteresis: propagates edges through weak pixels within each
 *        active tile until nothing changes.
 *
 * Only tiles marked in @p active do any work; the others return at once.
 * A tile that changed pixels on its border marks the neighbors those
 * pixels touch in @p next, the tiles to process in the next launch: a tile
 * can only have work left if a neighbor changed their shared border since
 * it last ran. The tile's 1-pixel border comes from the neighboring tiles
 * as they were when loaded; a neighbor changing it later in the same launch
 * marks the tile for the next one. The host relaunches until a launch
 * changes nothing, which is the unique fixed point: every weak pixel
 * connected to an edge is an edge, the same result as the CPU's flood fill.
 *
 * @param map       Class map (none / weak / edge), updated in place.
 * @param active    Tiles to process in this launch.
 * @param next      Tiles to process in the next launch (zeroed by the host).
 * @param changed   Set to 1 if any pixel changed (may be NULL).
 * @param processed Count of tiles processed (atomically incremented).
 */
__global__ void
canny_hyst_kernel(unsigned char *__restrict__ map, int w, int h,
                  const unsigned char *__restrict__ active, unsigned char *__restrict__ next,
                  int tiles_x, int tiles_y, int *__restrict__ changed, int *__restrict__ processed)
{
	__shared__ unsigned char s[HYST_T + 2][HYST_T + 2];
	__shared__ int border;
	const int bx = blockIdx.x, by = blockIdx.y;
	const int tx = threadIdx.x, ty = threadIdx.y;
	const int tid = ty * HYST_T + tx, nt = HYST_T * HYST_ROWS;
	const int x0 = bx * HYST_T - 1, y0 = by * HYST_T - 1;
	unsigned int mask = 0;
	int any = 0;

	/* the whole block decides together: no barrier has been reached yet */
	if (!active[by * tiles_x + bx])
		return;

	if (tid == 0) {
		border = 0;
		atomicAdd(processed, 1);
	}

	for (int i = tid; i < (HYST_T + 2) * (HYST_T + 2); i += nt) {
		const int r = i / (HYST_T + 2), c = i % (HYST_T + 2);
		const int gx = x0 + c, gy = y0 + r;

		s[r][c] = (gx >= 0 && gy >= 0 && gx < w && gy < h) ? map[(size_t)gy * w + gx] : CANNY_NONE;
	}
	__syncthreads();

	for (;;) {
		int ch = 0;

		for (int k = 0; k < HYST_T / HYST_ROWS; k++)
			if (hyst_relax(s, ty + k * HYST_ROWS + 1, tx + 1)) {
				mask |= 1u << k;
				ch = 1;
			}

		/* barrier that also tells every thread whether anyone changed */
		if (!__syncthreads_or(ch))
			break;
		any = 1;
	}

	/* write back the pixels that changed, noting changed border pixels */
	for (int k = 0; k < HYST_T / HYST_ROWS; k++) {
		const int r = ty + k * HYST_ROWS + 1, c = tx + 1;
		int bits;

		if (!(mask & (1u << k)))
			continue;

		map[(size_t)(y0 + r) * w + (x0 + c)] = CANNY_EDGE;
		bits = hyst_border_bits(r, c);
		if (bits)
			atomicOr(&border, bits);
	}
	__syncthreads();

	if (tid == 0) {
		if (any && changed)
			*changed = 1;
		hyst_mark(next, tiles_x, tiles_y, bx, by, border);
	}
}

/**
 * @brief Class map to edge map in place: 255 on edges, 0 elsewhere.
 */
__global__ void
canny_finish_kernel(unsigned char *__restrict__ map, size_t n)
{
	const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
		map[i] = (map[i] == CANNY_EDGE) ? 255 : 0;
}

/* ------------------------------------------------------------------------- */
/*                              Device Buffers                               */
/* ------------------------------------------------------------------------- */

/* Device buffers, reused between images and grown when needed. */
static unsigned char *d_pad = NULL, *d_out = NULL;
static float *d_tab = NULL;
static size_t cap_pad = 0, cap_out = 0, cap_tab = 0;

/* Edge detection buffers */
static unsigned char *e_img = NULL, *e_map = NULL, *e_act[2] = { NULL, NULL };
static int *e_hb = NULL, *e_b = NULL, *e_flag = NULL;
static size_t cap_eimg = 0, cap_emap = 0, cap_ehb = 0, cap_eb = 0, cap_eflag = 0;
static size_t cap_eact[2] = { 0, 0 };

/* Events bracketing the steps of one image */
#define NEV 5
static cudaEvent_t ev[NEV];
static int ev_ready = 0;

/*
 * Hysteresis launches per check of the "changed" flag. Reading the flag is
 * a synchronization with the GPU; launching a few kernels back to back and
 * checking only the last one saves most of them. Extra launches after the
 * fixed point is reached change nothing.
 */
#define HYST_GROUP 4

/**
 * @brief Reports a CUDA error. Returns 1 if @p err is an error.
 */
static int
cuda_failed(cudaError_t err, const char *what)
{
	if (err == cudaSuccess)
		return 0;
	uerrf("CUDA error in %s: %s", what, cudaGetErrorString(err));
	return 1;
}

/**
 * @brief Makes sure a device buffer holds at least @p need bytes.
 */
static int
reserve(void **buf, size_t *cap, size_t need, const char *what)
{
	if (need <= *cap)
		return 0;

	cudaFree(*buf);
	*buf = NULL;
	*cap = 0;

	if (cuda_failed(cudaMalloc(buf, need), what))
		return 1;

	*cap = need;
	return 0;
}

/* ------------------------------------------------------------------------- */
/*                            Public API Functions                           */
/* ------------------------------------------------------------------------- */

/**
 * @brief Initializes the first CUDA device.
 *
 * Makes the calling thread wait for the GPU by sleeping rather than spinning,
 * so that a thread waiting for the GPU leaves its core to the CPU stages.
 *
 * @param[out] info Device information (may be NULL).
 *
 * @return 0 on success, 1 if no usable GPU (already reported).
 */
extern "C" int
gpu_init(GpuInfo *info)
{
	cudaDeviceProp prop;
	int count = 0;

	if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
		uerrf("no CUDA device available");
		return 1;
	}

	if (cuda_failed(cudaSetDevice(0), "cudaSetDevice")
	    || cuda_failed(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync), "cudaSetDeviceFlags")
	    || cuda_failed(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties"))
		return 1;

	/* create the context now, so that its cost is not charged to an image */
	if (cuda_failed(cudaFree(0), "context creation"))
		return 1;

	/* blocking-sync events: waiting for them sleeps instead of spinning */
	for (int i = 0; i < NEV; i++)
		if (cuda_failed(cudaEventCreateWithFlags(&ev[i], cudaEventBlockingSync), "cudaEventCreate"))
			return 1;
	ev_ready = 1;

	if (info) {
		snprintf(info->name, sizeof(info->name), "%s", prop.name);
		info->cc_major = prop.major;
		info->cc_minor = prop.minor;
		info->sm_count = prop.multiProcessorCount;
		info->mem_gb = (double)prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0;
		cudaDriverGetVersion(&info->driver_version);
		cudaRuntimeGetVersion(&info->runtime_version);
	}

	return 0;
}

/**
 * @brief Releases the GPU buffers. Safe to call when not initialized.
 */
extern "C" void
gpu_shutdown(void)
{
	cudaFree(d_pad);
	cudaFree(d_out);
	cudaFree(d_tab);
	d_pad = d_out = NULL;
	d_tab = NULL;
	cap_pad = cap_out = cap_tab = 0;

	cudaFree(e_img);
	cudaFree(e_map);
	cudaFree(e_hb);
	cudaFree(e_b);
	cudaFree(e_flag);
	cudaFree(e_act[0]);
	cudaFree(e_act[1]);
	e_img = e_map = e_act[0] = e_act[1] = NULL;
	e_hb = e_b = e_flag = NULL;
	cap_eimg = cap_emap = cap_ehb = cap_eb = cap_eflag = 0;
	cap_eact[0] = cap_eact[1] = 0;

	if (ev_ready)
		for (int i = 0; i < NEV; i++)
			cudaEventDestroy(ev[i]);
	ev_ready = 0;
}

/**
 * @brief Denoises a prepared job on the GPU, writing its output image.
 *
 * Uploads the padded image and the weight table, runs the kernel and
 * downloads the result into the job's output. Device buffers are reused
 * between calls and grown when needed.
 *
 * @note Not thread-safe: call from one thread (the pipeline's GPU thread).
 *
 * With @p keep set, the result is left on the GPU for gpu_edges() instead
 * of being downloaded (the job's output buffer is then not filled).
 *
 * @param[in,out] job    Job prepared with nlm_job_prepare().
 * @param[in,out] timing GPU time of the steps, added to (may be NULL).
 * @param[in]     keep   Non-zero to keep the result on the GPU.
 *
 * @return IMG_OK, IMG_ERR_UNSUPPORTED if the job has no weight table, or
 *         IMG_ERR_GPU on a CUDA error (already reported).
 */
extern "C" int
gpu_denoise(NlmJob *job, GpuTiming *timing, int keep)
{
	const NlmContext *c = &job->ctx;
	const int r = c->p + c->s;
	const size_t pw = c->pw;
	const size_t ph = (size_t)c->h + 2 * r;
	const size_t npad = pw * ph;
	const size_t nout = (size_t)c->w * c->h;
	size_t ntab;
	cudaError_t err = cudaSuccess;
	dim3 block(BLOCK_W, BLOCK_H);
	dim3 grid((unsigned)((c->w + TILE_W - 1) / TILE_W), (unsigned)((c->h + TILE_H - 1) / TILE_H));
	size_t smem = tile_bytes(c->p, c->s);

	/* no noise: the output was already set when the job was prepared */
	if (nlm_job_bands(job) == 0)
		return IMG_OK;

	/* the GPU always reads weights from the table, and has kernels up to MAX_P */
	if (!c->wtab || c->p > MAX_P)
		return IMG_ERR_UNSUPPORTED;

	ntab = (size_t)c->cutoff + 2;

	if (reserve((void **)&d_pad, &cap_pad, npad, "cudaMalloc (image)")
	    || reserve((void **)&d_out, &cap_out, nout, "cudaMalloc (output)")
	    || reserve((void **)&d_tab, &cap_tab, ntab * sizeof(float), "cudaMalloc (weights)"))
		return IMG_ERR_GPU;

	cudaEventRecord(ev[0]);

	if (cuda_failed(cudaMemcpy(d_pad, c->pad, npad, cudaMemcpyHostToDevice), "upload (image)")
	    || cuda_failed(cudaMemcpy(d_tab, c->wtab, ntab * sizeof(float), cudaMemcpyHostToDevice),
	                   "upload (weights)"))
		return IMG_ERR_GPU;

	cudaEventRecord(ev[1]);

	/* the patch radius is a template parameter: pick its instantiation */
	switch (c->p) {
#define CASE(P) case P: err = launch<P>(grid, block, smem, d_pad, (int)pw, (int)ph, \
	                                (int)c->w, (int)c->h, c->s, c->cutoff, d_tab, d_out); break;
	CASE(0) CASE(1) CASE(2) CASE(3) CASE(4) CASE(5)
	CASE(6) CASE(7) CASE(8) CASE(9) CASE(10)
#undef CASE
	}

	cudaEventRecord(ev[2]);

	if (cuda_failed(err, "kernel launch"))
		return IMG_ERR_GPU;

	/* kept on the GPU for edge detection: no download */
	if (!keep && cuda_failed(cudaMemcpy(job->out.data, d_out, nout, cudaMemcpyDeviceToHost),
	                         "download (output)"))
		return IMG_ERR_GPU;

	cudaEventRecord(ev[3]);

	if (timing) {
		float up = 0.0f, kern = 0.0f, down = 0.0f;

		if (cuda_failed(cudaEventSynchronize(ev[3]), "cudaEventSynchronize"))
			return IMG_ERR_GPU;

		cudaEventElapsedTime(&up, ev[0], ev[1]);
		cudaEventElapsedTime(&kern, ev[1], ev[2]);
		cudaEventElapsedTime(&down, ev[2], ev[3]);
		timing->upload_s += up / 1000.0;
		timing->denoise_s += kern / 1000.0;
		timing->download_s += down / 1000.0;
	}

	return IMG_OK;
}

/**
 * @brief Detects the edges of an image on the GPU, replacing it with the
 *        edge map (255 on edges, 0 elsewhere).
 *
 * Uploads the image (or, with @p on_gpu set, uses the result gpu_denoise()
 * kept on the GPU), runs the blur, gradient/suppression and hysteresis
 * kernels, and downloads the edge map into the image's buffer. The result
 * is identical to the CPU's (canny_band() and canny_hysteresis()).
 *
 * @note Not thread-safe: call from one thread (the pipeline's GPU thread).
 *
 * @param[in,out] img    Image (its pixels are not read when @p on_gpu is set).
 * @param[in]     s      Setup from canny_setup().
 * @param[in,out] timing GPU time of the steps, added to (may be NULL).
 * @param[in]     on_gpu Non-zero to use the denoised image kept on the GPU.
 *
 * @return IMG_OK, or IMG_ERR_GPU on a CUDA error (already reported).
 */
extern "C" int
gpu_edges(Image *img, const CannySetup *s, GpuTiming *timing, int on_gpu)
{
	const int w = (int)img->width, h = (int)img->height;
	const size_t n = (size_t)w * h;
	dim3 block(CANNY_BW, CANNY_BH);
	dim3 grid((unsigned)((w + CANNY_BW - 1) / CANNY_BW), (unsigned)((h + CANNY_BH - 1) / CANNY_BH));
	dim3 hblock(HYST_T, HYST_ROWS);
	dim3 hgrid((unsigned)((w + HYST_T - 1) / HYST_T), (unsigned)((h + HYST_T - 1) / HYST_T));
	const int ntiles = (int)(hgrid.x * hgrid.y);
	const unsigned char *src;
	int flag, launches = 0, cur = 0, counts[2];

	/* e_flag holds the "changed" flag and the processed-tiles counter */
	if (reserve((void **)&e_img, &cap_eimg, n, "cudaMalloc (edges image)")
	    || reserve((void **)&e_map, &cap_emap, n, "cudaMalloc (edges map)")
	    || reserve((void **)&e_hb, &cap_ehb, n * sizeof(int), "cudaMalloc (edges blur)")
	    || reserve((void **)&e_b, &cap_eb, n * sizeof(int), "cudaMalloc (edges blur)")
	    || reserve((void **)&e_flag, &cap_eflag, 2 * sizeof(int), "cudaMalloc (edges flag)")
	    || reserve((void **)&e_act[0], &cap_eact[0], (size_t)ntiles, "cudaMalloc (active tiles)")
	    || reserve((void **)&e_act[1], &cap_eact[1], (size_t)ntiles, "cudaMalloc (active tiles)"))
		return IMG_ERR_GPU;

	if (cuda_failed(cudaMemcpyToSymbol(c_weights, s->weights, sizeof(int) * (2 * s->radius + 1)),
	                "upload (Gaussian weights)"))
		return IMG_ERR_GPU;

	cudaEventRecord(ev[0]);

	/* the denoised image is already on the GPU, in the denoising output */
	if (on_gpu) {
		src = d_out;
	} else {
		if (cuda_failed(cudaMemcpy(e_img, img->data, n, cudaMemcpyHostToDevice), "upload (edges image)"))
			return IMG_ERR_GPU;
		src = e_img;
	}

	cudaEventRecord(ev[1]);

	canny_blur_h_kernel<<<grid, block>>>(src, e_hb, w, h, s->radius);
	canny_blur_v_kernel<<<grid, block>>>(e_hb, e_b, w, h, s->radius);
	canny_classify_kernel<<<grid, block>>>(e_b, e_map, w, h, s->low2, s->high2);
	if (cuda_failed(cudaGetLastError(), "edge kernels"))
		return IMG_ERR_GPU;

	cudaEventRecord(ev[2]);

	/*
	 * Hysteresis: relaunch until a launch changes nothing, processing only
	 * the active tiles: all of them at first, then those whose neighbors
	 * changed their shared border. The flag is only checked on every
	 * HYST_GROUP-th launch (each check waits for the GPU); launches after
	 * convergence find no active tile and cost almost nothing.
	 */
	if (cuda_failed(cudaMemsetAsync(e_act[0], 1, (size_t)ntiles), "cudaMemset (active tiles)")
	    || cuda_failed(cudaMemsetAsync(e_flag, 0, 2 * sizeof(int)), "cudaMemset (edges flag)"))
		return IMG_ERR_GPU;

	do {
		for (int k = 0; k < HYST_GROUP; k++) {
			const int last = (k == HYST_GROUP - 1);

			cudaMemsetAsync(e_act[1 - cur], 0, (size_t)ntiles);
			if (last)
				cudaMemsetAsync(e_flag, 0, sizeof(int));

			canny_hyst_kernel<<<hgrid, hblock>>>(e_map, w, h, e_act[cur], e_act[1 - cur],
			                                     (int)hgrid.x, (int)hgrid.y,
			                                     last ? e_flag : NULL, e_flag + 1);
			cur = 1 - cur;
		}
		launches += HYST_GROUP;

		if (cuda_failed(cudaGetLastError(), "hysteresis kernel")
		    || cuda_failed(cudaMemcpy(&flag, e_flag, sizeof(int), cudaMemcpyDeviceToHost),
		                   "download (edges flag)"))
			return IMG_ERR_GPU;
	} while (flag);

	if (cuda_failed(cudaMemcpy(counts, e_flag, 2 * sizeof(int), cudaMemcpyDeviceToHost),
	                "download (tile count)"))
		return IMG_ERR_GPU;

	canny_finish_kernel<<<(unsigned)((n + 255) / 256), 256>>>(e_map, n);
	if (cuda_failed(cudaGetLastError(), "edge map kernel"))
		return IMG_ERR_GPU;

	cudaEventRecord(ev[3]);

	if (cuda_failed(cudaMemcpy(img->data, e_map, n, cudaMemcpyDeviceToHost), "download (edge map)"))
		return IMG_ERR_GPU;

	cudaEventRecord(ev[4]);

	if (timing) {
		float up = 0.0f, pix = 0.0f, hyst = 0.0f, down = 0.0f;

		if (cuda_failed(cudaEventSynchronize(ev[4]), "cudaEventSynchronize"))
			return IMG_ERR_GPU;

		cudaEventElapsedTime(&up, ev[0], ev[1]);
		cudaEventElapsedTime(&pix, ev[1], ev[2]);
		cudaEventElapsedTime(&hyst, ev[2], ev[3]);
		cudaEventElapsedTime(&down, ev[3], ev[4]);
		timing->upload_s += up / 1000.0;
		timing->edges_s += (pix + hyst) / 1000.0;
		timing->hysteresis_s += hyst / 1000.0;
		timing->download_s += down / 1000.0;
		timing->launches += launches;
		timing->tiles += counts[1];
		timing->tiles_all += (double)launches * ntiles;
	}

	return IMG_OK;
}
