/**
 * @file gpu.cu
 * @brief NLM denoising and Canny edge detection in CUDA.
 *
 * NLM: each block loads its image tile (plus the P + S border) into shared
 * memory once, then walks the search offsets GROUP at a time. All threads
 * compute the horizontal patch sums of the group straight from the tile;
 * after a barrier each thread slides a vertical window down its PIX_Y
 * pixels. Those are the CPU's exact integer patch sums, and the weights
 * come from the same nlm_weight() as the CPU's table, so with -fmad=false
 * the output is identical.
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

extern "C" {
#include "canny.h"
#include "error.h"
#include "nlm.h"
}
#include "gpu.h"

/*
 * Each thread computes PIX_Y vertically adjacent pixels. Taller tiles make
 * the 2P rows of extra horizontal sums a smaller share.
 */
#define BLOCK_W 32
#define BLOCK_H 16
#define PIX_Y   2
#define TILE_W  BLOCK_W
#define TILE_H  (BLOCK_H * PIX_Y)

/*
 * GROUP offsets per barrier round (4 S (S + 1) is a multiple of 8). Each
 * work item computes SEG horizontal sums, sliding over SEG + 2P squared
 * differences held in registers.
 */
#define GROUP 4
#define SEG   4
static_assert(SEG == 4, "tile_hsum stores its sums as one int4");

/* the program's largest patch radius */
#define MAX_P 10

struct Tile {
	unsigned char *img;
	int *hsum;          /* GROUP x dh x TILE_W, 16-byte aligned */
	int sw, sh;         /* image tile size */
	int pitch;          /* bytes per row of the image tile, see tile_pitch() */
	int s;
};

/*
 * Templates over P, so that the sliding window lives in registers and every
 * division is by a constant. They are also __host__, so that they can be
 * tested on the CPU, one thread after the other.
 */

/*
 * In tile_hsum, a warp reads 4 rows of the image tile, 8 words each. The
 * pitch makes those rows start 8 or 24 banks apart, so that they fall on
 * distinct banks: (pitch / 4) % 16 == 8.
 */
__host__ __device__ static inline int
tile_pitch(int sw)
{
	int words = (sw + 3) / 4;

	while (words % 16 != 8)
		words++;
	return 4 * words;
}

__host__ __device__ static inline size_t
tile_bytes(int p, int s)
{
	int r = p + s;
	size_t img = (size_t)tile_pitch(TILE_W + 2 * r) * (TILE_H + 2 * r);
	size_t hs = (size_t)GROUP * TILE_W * (TILE_H + 2 * p);

	img = (img + 15) & ~(size_t)15; /* align the int array for 16-byte stores */
	return img + hs * sizeof(int);
}

__host__ __device__ static inline Tile
tile_make(unsigned char *smem, int p, int s)
{
	Tile t;
	int r = p + s;
	size_t img;

	t.s = s;
	t.sw = TILE_W + 2 * r;
	t.sh = TILE_H + 2 * r;
	t.pitch = tile_pitch(t.sw);

	img = ((size_t)t.pitch * t.sh + 15) & ~(size_t)15;
	t.img = smem;
	t.hsum = (int *)(smem + img);
	return t;
}

/* Beyond the padded image (right/bottom blocks): zeros. */
__host__ __device__ static inline void
tile_load(const Tile *t, const unsigned char *pad, int pw, int ph,
          int x0, int y0, int tx, int ty)
{
	for (int r = ty; r < t->sh; r += BLOCK_H) {
		const int gr = y0 + r;

		for (int c = tx; c < t->sw; c += BLOCK_W) {
			const int gc = x0 + c;
			t->img[r * t->pitch + c] = (gr < ph && gc < pw) ? pad[(size_t)gr * pw + gc] : 0;
		}
	}
}

/* (*cx, *cy): running position in the search window, center skipped */
__host__ __device__ static inline void
next_offsets(int s, int pitch, int *cx, int *cy, int dx[GROUP], int dy[GROUP], int shift[GROUP])
{
	for (int k = 0; k < GROUP; k++) {
		if (*cx == 0 && *cy == 0) {
			if (++*cx > s) { *cx = -s; ++*cy; }
		}
		dx[k] = *cx;
		dy[k] = *cy;
		shift[k] = *cy * pitch + *cx;
		if (++*cx > s) { *cx = -s; ++*cy; }
	}
}

/* A work item: offset k, row r, SEG columns. Row r is image tile row r + S. */
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
		const unsigned char *a = t->img + (r + t->s) * t->pitch + (x0 + t->s);
		const unsigned char *b = a + shift[k];
		int *h = t->hsum + (k * dh + r) * TILE_W + x0;
		int d[SEG + 2 * P];
		int hv[SEG];
		int sum = 0;

		#pragma unroll
		for (int m = 0; m < SEG + 2 * P; m++) {
			const int v = (int)a[m] - (int)b[m];
			d[m] = v * v;
		}

		#pragma unroll
		for (int j = 0; j < side; j++)
			sum += d[j];
		hv[0] = sum;

		#pragma unroll
		for (int i = 1; i < SEG; i++) {
			sum += d[i + side - 1] - d[i - 1];
			hv[i] = sum;
		}

		/* one 16-byte store: a warp's four 4-word stores would conflict 4-way */
		*(int4 *)h = make_int4(hv[0], hv[1], hv[2], hv[3]);
	}
}

/* The thread's PIX_Y patch sums for offset k, sliding the window down. */
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

__host__ __device__ static inline void
accumulate(float w, unsigned char q, float *wsum, float *vsum, float *wmax)
{
	*wsum += w;
	*vsum += w * (float)q;
	if (w > *wmax)
		*wmax = w;
}

/* The center counts as much as its most similar neighbor, as on the CPU. */
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

/* Blocks per SM to keep: 3 x 512 threads (at most 42 registers), 2 on Turing. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#	define MIN_BLOCKS 2
#else
#	define MIN_BLOCKS 3
#endif

template <int P>
__global__ void __launch_bounds__(BLOCK_W * BLOCK_H, MIN_BLOCKS)
nlm_kernel(const unsigned char *__restrict__ pad, int pw, int ph, int w, int h,
           int s, int cutoff, float offset, float scale,
           unsigned char *__restrict__ out)
{
	extern __shared__ __align__(16) unsigned char smem[];
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

		next_offsets(s, t.pitch, &cx, &cy, dx, dy, shift);

		tile_hsum<P>(&t, shift, tid, nt);
		__syncthreads();

		for (int k = 0; k < GROUP; k++) {
			int ssd[PIX_Y];

			tile_patch<P>(&t, k, tx, ty, ssd);

			for (int j = 0; j < PIX_Y; j++)
				if (ssd[j] <= cutoff)
					accumulate(nlm_weight(ssd[j], offset, scale),
					           t.img[(ty * PIX_Y + j + r + dy[k]) * t.pitch + (tx + r + dx[k])],
					           &wsum[j], &vsum[j], &wmax[j]);
		}
		__syncthreads();                /* everyone done reading the sums */
	}

	for (int j = 0; j < PIX_Y; j++) {
		const int y = y0 + ty * PIX_Y + j, x = x0 + tx;

		if (x < w && y < h)
			out[(size_t)y * w + x] =
				finish(wsum[j], vsum[j], wmax[j], t.img[(ty * PIX_Y + j + r) * t.pitch + (tx + r)]);
	}
}

template <int P>
static cudaError_t
launch(dim3 grid, dim3 block, size_t smem, cudaStream_t stream, const unsigned char *pad,
       int pw, int ph, int w, int h, int s, int cutoff, float offset, float scale,
       unsigned char *out)
{
	if (smem > 48 * 1024) {
		cudaError_t err = cudaFuncSetAttribute(nlm_kernel<P>,
		                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
		                                       (int)smem);
		if (err != cudaSuccess)
			return err;
	}

	nlm_kernel<P><<<grid, block, smem, stream>>>(pad, pw, ph, w, h, s, cutoff, offset, scale, out);
	return cudaGetLastError();
}

/* Hysteresis tiles: HYST_T x HYST_T, HYST_T / HYST_ROWS rows per thread. */
#define CANNY_BW  32
#define CANNY_BH  8
#define HYST_T    32
#define HYST_ROWS 8

__constant__ int c_weights[2 * CANNY_MAX_RADIUS + 1];

__global__ void
canny_blur_h_kernel(const unsigned char *__restrict__ img, int *__restrict__ hb,
                    int w, int h, int r)
{
	const int x = blockIdx.x * CANNY_BW + threadIdx.x;
	const int y = blockIdx.y * CANNY_BH + threadIdx.y;

	if (x < w && y < h)
		hb[(size_t)y * w + x] = canny_blur_h(img, w, x, y, c_weights, r);
}

__global__ void
canny_blur_v_kernel(const int *__restrict__ hb, int *__restrict__ b, int w, int h, int r)
{
	const int x = blockIdx.x * CANNY_BW + threadIdx.x;
	const int y = blockIdx.y * CANNY_BH + threadIdx.y;

	if (x < w && y < h)
		b[(size_t)y * w + x] = canny_blur_v(hb, 0, w, h, x, y, c_weights, r);
}

__global__ void
canny_classify_kernel(const int *__restrict__ b, unsigned char *__restrict__ map,
                      int w, int h, long long tl2, long long th2)
{
	const int x = blockIdx.x * CANNY_BW + threadIdx.x;
	const int y = blockIdx.y * CANNY_BH + threadIdx.y;

	if (x < w && y < h)
		map[(size_t)y * w + x] = canny_classify(b, 0, w, h, x, y, tl2, th2);
}

/* (r, c) is 1-based inside the tile's halo. */
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

/* A changed border pixel affects the neighbor on that side (corner: diagonal). */
#define BORDER_TOP    0x01
#define BORDER_BOTTOM 0x02
#define BORDER_LEFT   0x04
#define BORDER_RIGHT  0x08
#define BORDER_TL     0x10
#define BORDER_TR     0x20
#define BORDER_BL     0x40
#define BORDER_BR     0x80

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

/*
 * Propagates edges through weak pixels within each active tile until
 * nothing changes. A tile can only have work left if a neighbor changed
 * their shared border, so only those are marked for the next launch. The
 * host relaunches until a launch changes nothing: the unique fixed point,
 * equal to the CPU's flood fill whatever the order.
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

	/* uniform per block, before any barrier */
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

__global__ void
canny_finish_kernel(unsigned char *__restrict__ map, size_t n)
{
	const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
		map[i] = (map[i] == CANNY_EDGE) ? 255 : 0;
}

/*
 * Two slots, each with its own stream, device buffers and pinned host
 * buffers: while one image's kernels run, the other slot uploads the next
 * image and downloads the previous one.
 */
#define SLOTS 2

/* Event pairs right around each piece of work, so waits between do not count. */
enum {
	EV_UP0, EV_UP1, EV_DN0, EV_DN1, EV_PX0, EV_PX1, EV_HY0, EV_HY1, EV_DL0, EV_DL1,
	EV_COUNT
};

/*
 * Checking the "changed" flag waits for the GPU, so it is checked every
 * HYST_GROUP launches; extra launches find no active tile and cost little.
 */
#define HYST_GROUP 4

struct Buf {
	void *p;
	size_t cap;
};

struct Slot {
	cudaStream_t stream;
	cudaEvent_t ev[EV_COUNT];
	Buf d_pad, d_out;
	Buf e_img, e_hb, e_b, e_map, e_act[2];
	Buf e_flag;                            /* changed flag, tile counter */
	Buf h_in, h_out, h_flag;               /* pinned */
	GpuTask *task;
	int denoise, edges;
	int launches;
	double tiles_all;
	double hyst_ms;
	int cur;                               /* active-tile array of the next launch */
	int tiles_x, tiles_y;
};

static Slot slots[SLOTS];
static int ready = 0;

static int
cuda_failed(cudaError_t err, const char *what)
{
	if (err == cudaSuccess)
		return 0;
	uerrf("CUDA error in %s: %s", what, cudaGetErrorString(err));
	return 1;
}

static int
dev_reserve(Buf *b, size_t need, const char *what)
{
	if (need <= b->cap)
		return 0;

	cudaFree(b->p);
	b->p = NULL;
	b->cap = 0;

	if (cuda_failed(cudaMalloc(&b->p, need), what))
		return 1;

	b->cap = need;
	return 0;
}

static int
host_reserve(Buf *b, size_t need, const char *what)
{
	if (need <= b->cap)
		return 0;

	cudaFreeHost(b->p);
	b->p = NULL;
	b->cap = 0;

	if (cuda_failed(cudaHostAlloc(&b->p, need, cudaHostAllocDefault), what))
		return 1;

	b->cap = need;
	return 0;
}

/* From the image: a job with nothing to denoise leaves its context unset. */
static void
task_size(const GpuTask *t, int *w, int *h)
{
	*w = (int)t->img->width;
	*h = (int)t->img->height;
}

static int
slot_stage(Slot *s, GpuTask *t, int edges)
{
	const NlmContext *c = t->job ? &t->job->ctx : NULL;
	int w, h;
	size_t n;

	task_size(t, &w, &h);
	n = (size_t)w * h;

	s->task = t;
	s->denoise = (t->job && nlm_job_bands(t->job) > 0);
	s->edges = edges;
	s->launches = 0;
	s->tiles_all = 0.0;
	s->hyst_ms = 0.0;

	if (s->denoise) {
		const size_t npad = c->pw * ((size_t)c->h + 2 * (c->p + c->s));

		if (host_reserve(&s->h_in, npad, "cudaHostAlloc (image)")
		    || dev_reserve(&s->d_pad, npad, "cudaMalloc (image)")
		    || dev_reserve(&s->d_out, n, "cudaMalloc (output)"))
			return IMG_ERR_GPU;

		memcpy(s->h_in.p, c->pad, npad);

		cudaEventRecord(s->ev[EV_UP0], s->stream);
		if (cuda_failed(cudaMemcpyAsync(s->d_pad.p, s->h_in.p, npad, cudaMemcpyHostToDevice,
		                                s->stream), "upload (image)"))
			return IMG_ERR_GPU;
	} else if (edges) {
		/* the image, or the output of a job with nothing to denoise */
		const unsigned char *src = t->job ? t->job->out.data : t->img->data;

		if (host_reserve(&s->h_in, n, "cudaHostAlloc (image)")
		    || dev_reserve(&s->e_img, n, "cudaMalloc (edges image)"))
			return IMG_ERR_GPU;

		memcpy(s->h_in.p, src, n);

		cudaEventRecord(s->ev[EV_UP0], s->stream);
		if (cuda_failed(cudaMemcpyAsync(s->e_img.p, s->h_in.p, n, cudaMemcpyHostToDevice,
		                                s->stream), "upload (edges image)"))
			return IMG_ERR_GPU;
	}

	cudaEventRecord(s->ev[EV_UP1], s->stream);
	return IMG_OK;
}

static int
slot_kernels(Slot *s, const CannySetup *cs)
{
	const GpuTask *t = s->task;
	int w, h;
	size_t n;

	task_size(t, &w, &h);
	n = (size_t)w * h;

	if (s->denoise) {
		const NlmContext *c = &t->job->ctx;
		const int r = c->p + c->s;
		dim3 block(BLOCK_W, BLOCK_H);
		dim3 grid((unsigned)((c->w + TILE_W - 1) / TILE_W), (unsigned)((c->h + TILE_H - 1) / TILE_H));
		size_t smem = tile_bytes(c->p, c->s);
		cudaError_t err = cudaSuccess;

		cudaEventRecord(s->ev[EV_DN0], s->stream);

		switch (c->p) {
#define CASE(P) case P: err = launch<P>(grid, block, smem, s->stream, (unsigned char *)s->d_pad.p, \
		                                (int)c->pw, (int)c->h + 2 * r, (int)c->w, (int)c->h, c->s, \
		                                c->cutoff, c->off_f, c->scale_f, (unsigned char *)s->d_out.p); break;
		CASE(0) CASE(1) CASE(2) CASE(3) CASE(4) CASE(5)
		CASE(6) CASE(7) CASE(8) CASE(9) CASE(10)
#undef CASE
		}

		if (cuda_failed(err, "denoising kernel"))
			return IMG_ERR_GPU;

		cudaEventRecord(s->ev[EV_DN1], s->stream);
	}

	if (s->edges) {
		dim3 block(CANNY_BW, CANNY_BH);
		dim3 grid((unsigned)((w + CANNY_BW - 1) / CANNY_BW), (unsigned)((h + CANNY_BH - 1) / CANNY_BH));
		const unsigned char *src = s->denoise ? (unsigned char *)s->d_out.p : (unsigned char *)s->e_img.p;

		if (dev_reserve(&s->e_hb, n * sizeof(int), "cudaMalloc (edges blur)")
		    || dev_reserve(&s->e_b, n * sizeof(int), "cudaMalloc (edges blur)")
		    || dev_reserve(&s->e_map, n, "cudaMalloc (edges map)"))
			return IMG_ERR_GPU;

		cudaEventRecord(s->ev[EV_PX0], s->stream);
		canny_blur_h_kernel<<<grid, block, 0, s->stream>>>(src, (int *)s->e_hb.p, w, h, cs->radius);
		canny_blur_v_kernel<<<grid, block, 0, s->stream>>>((int *)s->e_hb.p, (int *)s->e_b.p, w, h, cs->radius);
		canny_classify_kernel<<<grid, block, 0, s->stream>>>((int *)s->e_b.p, (unsigned char *)s->e_map.p,
		                                                    w, h, cs->low2, cs->high2);
		if (cuda_failed(cudaGetLastError(), "edge kernels"))
			return IMG_ERR_GPU;

		cudaEventRecord(s->ev[EV_PX1], s->stream);
	}

	return IMG_OK;
}

static int
hyst_group(Slot *s, int w, int h)
{
	const int ntiles = s->tiles_x * s->tiles_y;

	cudaEventRecord(s->ev[EV_HY0], s->stream);

	for (int k = 0; k < HYST_GROUP; k++) {
		const int last = (k == HYST_GROUP - 1);

		cudaMemsetAsync(s->e_act[1 - s->cur].p, 0, (size_t)ntiles, s->stream);
		if (last)
			cudaMemsetAsync(s->e_flag.p, 0, sizeof(int), s->stream);

		canny_hyst_kernel<<<dim3(s->tiles_x, s->tiles_y), dim3(HYST_T, HYST_ROWS), 0, s->stream>>>(
			(unsigned char *)s->e_map.p, w, h, (unsigned char *)s->e_act[s->cur].p,
			(unsigned char *)s->e_act[1 - s->cur].p, s->tiles_x, s->tiles_y,
			last ? (int *)s->e_flag.p : NULL, (int *)s->e_flag.p + 1);
		s->cur = 1 - s->cur;
	}

	s->launches += HYST_GROUP;
	s->tiles_all += (double)HYST_GROUP * ntiles;

	if (cuda_failed(cudaGetLastError(), "hysteresis kernel")
	    || cuda_failed(cudaMemcpyAsync(s->h_flag.p, s->e_flag.p, 2 * sizeof(int), cudaMemcpyDeviceToHost,
	                                   s->stream), "download (edges flag)"))
		return IMG_ERR_GPU;

	cudaEventRecord(s->ev[EV_HY1], s->stream);
	return IMG_OK;
}

/*
 * The first launch processes every tile and is the most expensive, so it is
 * started before the host copies the neighboring images.
 */
static int
slot_hyst_start(Slot *s)
{
	int w, h, ntiles;

	if (!s->edges)
		return IMG_OK;

	task_size(s->task, &w, &h);
	s->tiles_x = (w + HYST_T - 1) / HYST_T;
	s->tiles_y = (h + HYST_T - 1) / HYST_T;
	ntiles = s->tiles_x * s->tiles_y;
	s->cur = 0;

	/* e_flag holds the "changed" flag and the processed-tiles counter */
	if (dev_reserve(&s->e_flag, 2 * sizeof(int), "cudaMalloc (edges flag)")
	    || dev_reserve(&s->e_act[0], (size_t)ntiles, "cudaMalloc (active tiles)")
	    || dev_reserve(&s->e_act[1], (size_t)ntiles, "cudaMalloc (active tiles)")
	    || host_reserve(&s->h_flag, 2 * sizeof(int), "cudaHostAlloc (edges flag)"))
		return IMG_ERR_GPU;

	cudaMemsetAsync(s->e_act[0].p, 1, (size_t)ntiles, s->stream);
	cudaMemsetAsync(s->e_flag.p, 0, 2 * sizeof(int), s->stream);

	return hyst_group(s, w, h);
}

static int
slot_hyst_finish(Slot *s)
{
	int w, h;

	if (!s->edges)
		return IMG_OK;

	task_size(s->task, &w, &h);

	for (;;) {
		float ms = 0.0f;

		if (cuda_failed(cudaStreamSynchronize(s->stream), "cudaStreamSynchronize"))
			return IMG_ERR_GPU;

		cudaEventElapsedTime(&ms, s->ev[EV_HY0], s->ev[EV_HY1]);
		s->hyst_ms += ms;

		if (!((int *)s->h_flag.p)[0])
			return IMG_OK;

		if (hyst_group(s, w, h) != IMG_OK)
			return IMG_ERR_GPU;
	}
}

static int
slot_download(Slot *s)
{
	int w, h;
	size_t n;

	task_size(s->task, &w, &h);
	n = (size_t)w * h;

	if (s->denoise || s->edges) {
		const void *res = s->edges ? s->e_map.p : s->d_out.p;

		if (host_reserve(&s->h_out, n, "cudaHostAlloc (output)"))
			return IMG_ERR_GPU;

		cudaEventRecord(s->ev[EV_DL0], s->stream);

		if (s->edges) {
			canny_finish_kernel<<<(unsigned)((n + 255) / 256), 256, 0, s->stream>>>(
				(unsigned char *)s->e_map.p, n);
			if (cuda_failed(cudaGetLastError(), "edge map kernel"))
				return IMG_ERR_GPU;
		}

		if (cuda_failed(cudaMemcpyAsync(s->h_out.p, res, n, cudaMemcpyDeviceToHost, s->stream),
		                "download (output)"))
			return IMG_ERR_GPU;

		cudaEventRecord(s->ev[EV_DL1], s->stream);
	}

	return IMG_OK;
}

static void
slot_finish(Slot *s, int status, GpuTiming *timing, void (*done)(GpuTask *, void *), void *ctx)
{
	GpuTask *t = s->task;
	float up = 0.0f, dn = 0.0f, px = 0.0f, dl = 0.0f;

	if (!t)
		return;

	if (status == IMG_OK && (s->denoise || s->edges)
	    && cuda_failed(cudaEventSynchronize(s->ev[EV_DL1]), "cudaEventSynchronize"))
		status = IMG_ERR_GPU;

	if (status == IMG_OK) {
		int w, h;

		task_size(t, &w, &h);
		if (s->edges)
			memcpy(t->img->data, s->h_out.p, (size_t)w * h);
		else if (s->denoise)
			memcpy(t->job->out.data, s->h_out.p, (size_t)w * h);

		if (s->denoise || s->edges) {
			cudaEventElapsedTime(&up, s->ev[EV_UP0], s->ev[EV_UP1]);
			cudaEventElapsedTime(&dl, s->ev[EV_DL0], s->ev[EV_DL1]);
		}
		if (s->denoise)
			cudaEventElapsedTime(&dn, s->ev[EV_DN0], s->ev[EV_DN1]);
		if (s->edges)
			cudaEventElapsedTime(&px, s->ev[EV_PX0], s->ev[EV_PX1]);

		/* each filter with its share of the copies */
		t->denoise_s = s->denoise ? (up + dn + (s->edges ? 0.0 : dl)) / 1000.0 : 0.0;
		t->edges_s = s->edges ? ((s->denoise ? 0.0 : up) + px + s->hyst_ms + dl) / 1000.0 : 0.0;

		if (timing) {
			int *hf = (int *)s->h_flag.p;

			timing->upload_s += up / 1000.0;
			timing->denoise_s += dn / 1000.0;
			timing->edges_s += (px + s->hyst_ms) / 1000.0;
			timing->hysteresis_s += s->hyst_ms / 1000.0;
			timing->download_s += dl / 1000.0;
			if (s->edges) {
				timing->launches += s->launches;
				timing->tiles += hf[1];
				timing->tiles_all += s->tiles_all;
			}
		}
	}

	t->status = status;
	s->task = NULL;
	if (done)
		done(t, ctx);
}

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

	memset(slots, 0, sizeof(slots));
	for (int i = 0; i < SLOTS; i++) {
		if (cuda_failed(cudaStreamCreateWithFlags(&slots[i].stream, cudaStreamNonBlocking),
		                "cudaStreamCreate"))
			return 1;
		/* waiting sleeps instead of spinning, leaving the core to the CPU stages */
		for (int k = 0; k < EV_COUNT; k++)
			if (cuda_failed(cudaEventCreateWithFlags(&slots[i].ev[k], cudaEventBlockingSync),
			                "cudaEventCreate"))
				return 1;
	}
	ready = 1;

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

extern "C" void
gpu_shutdown(void)
{
	if (!ready)
		return;

	for (int i = 0; i < SLOTS; i++) {
		Slot *s = &slots[i];
		Buf *dev[] = { &s->d_pad, &s->d_out, &s->e_img, &s->e_hb, &s->e_b,
		               &s->e_map, &s->e_act[0], &s->e_act[1], &s->e_flag };
		Buf *host[] = { &s->h_in, &s->h_out, &s->h_flag };

		cudaStreamSynchronize(s->stream);
		for (size_t k = 0; k < sizeof(dev) / sizeof(dev[0]); k++)
			cudaFree(dev[k]->p);
		for (size_t k = 0; k < sizeof(host) / sizeof(host[0]); k++)
			cudaFreeHost(host[k]->p);
		for (int k = 0; k < EV_COUNT; k++)
			cudaEventDestroy(s->ev[k]);
		cudaStreamDestroy(s->stream);
	}

	memset(slots, 0, sizeof(slots));
	ready = 0;
}

extern "C" size_t
gpu_run(GpuTask *tasks, size_t n, const CannySetup *cs, GpuTiming *timing,
        void (*done)(GpuTask *, void *), void *ctx)
{
	size_t failed = 0, k = 0, cur = 0;
	int err = IMG_OK;

	/* the same Gaussian weights for every image */
	if (cs && cuda_failed(cudaMemcpyToSymbol(c_weights, cs->weights,
	                                         sizeof(int) * (2 * cs->radius + 1)),
	                      "upload (Gaussian weights)"))
		err = IMG_ERR_GPU;

	for (size_t i = 0; i < n; i++) {
		GpuTask *t = &tasks[i];
		const int dn = (t->job && nlm_job_bands(t->job) > 0);

		t->status = GPU_PENDING;
		t->denoise_s = t->edges_s = 0.0;

		if (dn && t->job->ctx.p > MAX_P)
			t->status = IMG_ERR_UNSUPPORTED;
		else if (!dn && !cs)
			t->status = IMG_OK;  /* denoising only, no noise: output already set */

		if (t->status != GPU_PENDING && done)
			done(t, ctx);
	}

#define NEXT(k) do { while ((k) < n && tasks[(k)].status != GPU_PENDING) (k)++; } while (0)

	NEXT(k);
	if (k < n && err == IMG_OK)
		err = slot_stage(&slots[cur], &tasks[k], cs != NULL);

	while (k < n && err == IMG_OK) {
		Slot *s = &slots[cur], *o = &slots[1 - cur];
		size_t next = k + 1;

		/* this image's kernels and its first hysteresis group, asynchronously */
		err = slot_kernels(s, cs);
		if (err == IMG_OK)
			err = slot_hyst_start(s);

		/* meanwhile, free the other slot and stage the next image there */
		NEXT(next);
		if (o->task)
			slot_finish(o, IMG_OK, timing, done, ctx);
		if (err == IMG_OK && next < n)
			err = slot_stage(o, &tasks[next], cs != NULL);

		if (err == IMG_OK)
			err = slot_hyst_finish(s);
		if (err == IMG_OK)
			err = slot_download(s);

		if (err != IMG_OK)
			break;

		k = next;
		cur = 1 - cur;
	}
#undef NEXT

	/* on an error, fail what is in flight and everything left */
	for (int i = 0; i < SLOTS; i++)
		if (slots[(cur + i) % SLOTS].task)
			slot_finish(&slots[(cur + i) % SLOTS], err == IMG_OK ? IMG_OK : IMG_ERR_GPU,
			            timing, done, ctx);

	for (size_t i = 0; i < n; i++) {
		if (tasks[i].status == GPU_PENDING) {
			tasks[i].status = IMG_ERR_GPU;
			if (done)
				done(&tasks[i], ctx);
		}
		failed += (tasks[i].status == IMG_ERR_GPU);
	}

	return failed;
}
