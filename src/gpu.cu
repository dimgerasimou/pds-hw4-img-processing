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
 * once. It then loops over all offsets of the search window; for each
 * offset it computes, in shared memory, the squared differences between
 * the tile and its shifted copy, sums them horizontally over the patch
 * width, and every thread sums its column over the patch height, sliding
 * the window down from one of its pixels to the next. That is each pixel's
 * exact integer patch sum, the same value the CPU obtains from integral
 * images. Weights are read from the table built by the CPU, and every
 * thread keeps its accumulators in registers for the whole loop, so they
 * cost no memory traffic. No integer division is used in the loops.
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
 * pixels. Taller tiles make the patch border (2P rows of extra differences
 * and sums) a smaller share of the work.
 */
#define BLOCK_W 32
#define BLOCK_H 16
#define PIX_Y   2
#define TILE_W  BLOCK_W
#define TILE_H  (BLOCK_H * PIX_Y)

/**
 * @struct Tile
 * @brief Shared-memory layout and geometry of one block.
 */
struct Tile {
	unsigned char *img; /**< Padded image tile: sh x sw bytes */
	int *diff;          /**< Squared differences: dh x dw */
	int *hsum;          /**< Horizontal patch sums: dh x TILE_W */
	int sw, sh;         /**< Image tile size (tile + 2 (P + S)) */
	int dw, dh;         /**< Difference tile size (tile + 2 P) */
	int p, s;           /**< Patch and search radii */
};

/* ------------------------------------------------------------------------- */
/*                               Kernel Phases                               */
/* ------------------------------------------------------------------------- */

/*
 * The phases take the thread's position (tx, ty) in the block and cover
 * their 2D arrays by stepping BLOCK_W columns and BLOCK_H rows at a time,
 * which needs no integer division (a multi-instruction routine on the GPU).
 */

/**
 * @brief Bytes of shared memory needed by one block.
 */
__host__ __device__ static inline size_t
tile_bytes(int p, int s)
{
	int r = p + s;
	size_t img = (size_t)(TILE_W + 2 * r) * (TILE_H + 2 * r);
	size_t dif = (size_t)(TILE_W + 2 * p) * (TILE_H + 2 * p);
	size_t hs = (size_t)TILE_W * (TILE_H + 2 * p);

	img = (img + 3) & ~(size_t)3; /* align the int arrays */
	return img + (dif + hs) * sizeof(int);
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

	t.p = p;
	t.s = s;
	t.sw = TILE_W + 2 * r;
	t.sh = TILE_H + 2 * r;
	t.dw = TILE_W + 2 * p;
	t.dh = TILE_H + 2 * p;

	img = ((size_t)t.sw * t.sh + 3) & ~(size_t)3;
	t.img = smem;
	t.diff = (int *)(smem + img);
	t.hsum = t.diff + t.dw * t.dh;
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
 * @brief Phase 2: squared differences between the tile and its copy
 *        shifted by (dx, dy), over the tile plus the patch border.
 *
 * Difference (r, c) belongs to image tile position (r + S, c + S).
 */
__host__ __device__ static inline void
tile_diff(const Tile *t, int dx, int dy, int tx, int ty)
{
	const int shift = dy * t->sw + dx;

	for (int r = ty; r < t->dh; r += BLOCK_H) {
		const unsigned char *a = t->img + (r + t->s) * t->sw + t->s;
		int *d = t->diff + r * t->dw;

		for (int c = tx; c < t->dw; c += BLOCK_W) {
			int v = (int)a[c] - (int)a[c + shift];
			d[c] = v * v;
		}
	}
}

/**
 * @brief Phase 3: sums of 2P+1 horizontally adjacent differences.
 */
__host__ __device__ static inline void
tile_hsum(const Tile *t, int tx, int ty)
{
	const int side = 2 * t->p + 1;

	for (int r = ty; r < t->dh; r += BLOCK_H) {
		const int *d = t->diff + r * t->dw + tx;
		int sum = 0;

		for (int j = 0; j < side; j++)
			sum += d[j];
		t->hsum[r * TILE_W + tx] = sum;
	}
}

/**
 * @brief Phase 4 (per thread): the patch sums of the thread's PIX_Y pixels,
 *        rows ty * PIX_Y ... of column tx, from 2P+1 vertically adjacent
 *        horizontal sums. The window slides down one row per pixel.
 */
__host__ __device__ static inline void
tile_patch(const Tile *t, int tx, int ty, int ssd[PIX_Y])
{
	const int side = 2 * t->p + 1;
	const int *h = t->hsum + (ty * PIX_Y) * TILE_W + tx;
	int sum = 0;

	for (int i = 0; i < side; i++)
		sum += h[i * TILE_W];
	ssd[0] = sum;

	for (int k = 1; k < PIX_Y; k++) {
		sum += h[(k - 1 + side) * TILE_W] - h[(k - 1) * TILE_W];
		ssd[k] = sum;
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
 * Offsets are visited in raster order, skipping the center, the same
 * order as on the CPU.
 *
 * @param pad    Mirrored, padded image ((H + 2R) x pw).
 * @param pw     Padded width.
 * @param ph     Padded height.
 * @param w      Image width.
 * @param h      Image height.
 * @param p      Patch radius.
 * @param s      Search radius.
 * @param cutoff Largest patch sum with a non-negligible weight.
 * @param wtab   Weight per patch sum 0..cutoff.
 * @param out    Output image (w x h).
 */
__global__ void
nlm_kernel(const unsigned char *__restrict__ pad, int pw, int ph, int w, int h,
           int p, int s, int cutoff, const float *__restrict__ wtab,
           unsigned char *__restrict__ out)
{
	extern __shared__ unsigned char smem[];
	const Tile t = tile_make(smem, p, s);
	const int tx = threadIdx.x, ty = threadIdx.y;
	const int x0 = blockIdx.x * TILE_W, y0 = blockIdx.y * TILE_H;
	const int r = p + s;
	float wsum[PIX_Y], vsum[PIX_Y], wmax[PIX_Y];

	for (int k = 0; k < PIX_Y; k++)
		wsum[k] = vsum[k] = wmax[k] = 0.0f;

	tile_load(&t, pad, pw, ph, x0, y0, tx, ty);
	__syncthreads();

	for (int dy = -s; dy <= s; dy++) {
		for (int dx = -s; dx <= s; dx++) {
			int ssd[PIX_Y];

			if (dx == 0 && dy == 0)
				continue;

			tile_diff(&t, dx, dy, tx, ty);
			__syncthreads();            /* differences complete */

			tile_hsum(&t, tx, ty);
			__syncthreads();            /* horizontal sums complete */

			tile_patch(&t, tx, ty, ssd);

			for (int k = 0; k < PIX_Y; k++)
				if (ssd[k] <= cutoff)
					accumulate(__ldg(&wtab[ssd[k]]),
					           t.img[(ty * PIX_Y + k + r + dy) * t.sw + (tx + r + dx)],
					           &wsum[k], &vsum[k], &wmax[k]);

			/*
			 * No barrier needed here: the next offset's differences
			 * overwrite only diff, which nobody reads any more, and the
			 * barrier after them keeps hsum intact until every thread has
			 * read it.
			 */
		}
	}

	for (int k = 0; k < PIX_Y; k++) {
		const int y = y0 + ty * PIX_Y + k, x = x0 + tx;

		if (x < w && y < h)
			out[(size_t)y * w + x] =
				finish(wsum[k], vsum[k], wmax[k], t.img[(ty * PIX_Y + k + r) * t.sw + (tx + r)]);
	}
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
	dim3 block(BLOCK_W, BLOCK_H);
	dim3 grid((unsigned)((c->w + TILE_W - 1) / TILE_W), (unsigned)((c->h + TILE_H - 1) / TILE_H));
	size_t smem = tile_bytes(c->p, c->s);

	/* no noise: the output was already set when the job was prepared */
	if (nlm_job_bands(job) == 0)
		return IMG_OK;

	/* the GPU always reads weights from the table */
	if (!c->wtab)
		return IMG_ERR_UNSUPPORTED;

	ntab = (size_t)c->cutoff + 2;

	if (reserve((void **)&d_pad, &cap_pad, npad, "cudaMalloc (image)")
	    || reserve((void **)&d_out, &cap_out, nout, "cudaMalloc (output)")
	    || reserve((void **)&d_tab, &cap_tab, ntab * sizeof(float), "cudaMalloc (weights)"))
		return IMG_ERR_GPU;

	if (smem > 48 * 1024
	    && cuda_failed(cudaFuncSetAttribute(nlm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
	                                        (int)smem), "cudaFuncSetAttribute"))
		return IMG_ERR_GPU;

	cudaEventRecord(ev[0]);

	if (cuda_failed(cudaMemcpy(d_pad, c->pad, npad, cudaMemcpyHostToDevice), "upload (image)")
	    || cuda_failed(cudaMemcpy(d_tab, c->wtab, ntab * sizeof(float), cudaMemcpyHostToDevice),
	                   "upload (weights)"))
		return IMG_ERR_GPU;

	cudaEventRecord(ev[1]);
	nlm_kernel<<<grid, block, smem>>>(d_pad, (int)pw, (int)ph, (int)c->w, (int)c->h,
	                                  c->p, c->s, c->cutoff, d_tab, d_out);
	cudaEventRecord(ev[2]);

	if (cuda_failed(cudaGetLastError(), "kernel launch")
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
