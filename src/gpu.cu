/**
 * @file gpu.cu
 * @brief CUDA implementation of NLM denoising.
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
/*                              Device Buffers                               */
/* ------------------------------------------------------------------------- */

/* Device buffers, reused between images and grown when needed. */
static unsigned char *d_pad = NULL, *d_out = NULL;
static float *d_tab = NULL;
static size_t cap_pad = 0, cap_out = 0, cap_tab = 0;

/* Events bracketing upload, kernel and download of one image */
static cudaEvent_t ev[4];
static int ev_ready = 0;

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
	for (int i = 0; i < 4; i++)
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

	if (ev_ready)
		for (int i = 0; i < 4; i++)
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
 * @param[in,out] job    Job prepared with nlm_job_prepare().
 * @param[in,out] timing GPU time of the three steps, added to (may be NULL).
 *
 * @return IMG_OK, IMG_ERR_UNSUPPORTED if the job has no weight table, or
 *         IMG_ERR_GPU on a CUDA error (already reported).
 */
extern "C" int
gpu_denoise(NlmJob *job, GpuTiming *timing)
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

	if (cuda_failed(err, "kernel launch")
	    || cuda_failed(cudaMemcpy(job->out.data, d_out, nout, cudaMemcpyDeviceToHost),
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
		timing->kernel_s += kern / 1000.0;
		timing->download_s += down / 1000.0;
	}

	return IMG_OK;
}
