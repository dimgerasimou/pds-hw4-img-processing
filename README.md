# Parallel Image Filters — CUDA and OpenMP Implementations

<p align="center">
  <img src="https://img.shields.io/badge/parallelism-OpenMP-blue" alt="OpenMP">
  <img src="https://img.shields.io/badge/acceleration-CUDA-green" alt="CUDA">
</p>

Assignment #4 of the **Parallel and Distributed Systems** coursework: [parallel-distributed-systems](https://github.com/dimgerasimou/parallel-distributed-systems)

**`imgfilter`** is a batch **image processing tool** implementing **Non-Local Means (NLM) denoising** and
**Canny edge detection** on multicore CPUs (**OpenMP**) and **single NVIDIA GPUs** (**CUDA**), with a strong
emphasis on comparing parallelization strategies and reproducible benchmarking.

> **Status:** the I/O pipeline, benchmarking, NLM denoising and Canny edge detection, on the CPU
> (OpenMP) and the GPU (CUDA), are implemented.

## Overview

This project studies how per-pixel image filters parallelize on CPUs and GPUs.

The main objectives are:
- Implement NLM denoising and Canny edge detection as a sequential C baseline
- Parallelize them with OpenMP and accelerate them with CUDA
- Compare parallelizing *within* an image (per pixel) against *across* images
- Measure every stage of the pipeline (I/O and compute) separately

Typical use cases are the preprocessing of large image sets, such as denoising low-dose CT or X-ray scans
and extracting edges before further analysis, where the number of images makes a serial implementation
impractical.

## Features

- Command-line tool with `cp`-like input/output semantics (file or directory in, file or directory out)
- Reads **PGM, PNG, JPEG, BMP and TGA** (converted to 8-bit grayscale); writes **PGM or PNG**
- Pipeline split into **read → decode → encode → write** stages, each parallelized across images with
  OpenMP and timed separately, so file I/O and (de)compression are never mixed in one measurement
- **Batch processing** with bounded memory: peak usage depends on the batch size, not the dataset size
- Configurable thread count
- Thread-safe **progress bars** for every stage
- Automated benchmarking with **JSON output**: system information, dataset information, and per-stage
  wall time, throughput and per-image statistics
- **Non-Local Means denoising** with integral images, parallelized over bands of all images of a batch
- **Canny edge detection**, with integer arithmetic shared by CPU and GPU
- **GPU filters** (CUDA), pipelined with the CPU stages, bit-for-bit identical to the CPU

## Build

### Requirements
- C compiler with OpenMP support (`gcc` or `clang`)
- `make`
- Optional: CUDA Toolkit and an NVIDIA GPU, for `-g`

### Compile
```bash
make                 # CUDA is used when nvcc is found
make CUDA=0          # CPU-only build
make GPU_ARCH=86     # CUDA architecture (default: detected with nvidia-smi, else 75)
```

Produces:
```
bin/imgfilter
```

Without a CUDA toolkit the program builds CPU-only and `-g` reports that CUDA is unavailable. Run
`make help` for all targets and overrides.

## Usage

```bash
./bin/imgfilter [options] <input>
```

| Option        | Description                                             |
| ------------- | ------------------------------------------------------- |
| `-o <output>` | Output file or directory (nothing is written if omitted) |
| `-f <format>` | Output format: `pgm` or `png` (see below)               |
| `-t <n>`      | Number of threads (default: all cores)                  |
| `-B <n>`      | Images per batch, `0` for all at once (default: 256)    |
| `-b <file>`   | Write benchmark results as JSON (`-` for stdout)        |
| `-n <n>`      | Timed benchmark trials (default: 1, requires `-b`)      |
| `-w <n>`      | Warmup benchmark trials (default: 0, requires `-b`)     |
| `-p`          | Show progress bars (on stderr)                          |
| `-d`          | Denoise with Non-Local Means (see below)                |
| `-e`          | Detect edges with Canny; the output is the edge map (see below) |
| `-g`          | Run the filters on the GPU (CUDA), pipelined with the CPU stages |
| `-h`          | Show help                                               |

Input and output follow the conventions of `cp`:

| Input     | Output         | Result                                              |
| --------- | -------------- | --------------------------------------------------- |
| file      | —              | Process only, nothing written                       |
| file      | file           | Write to that file                                  |
| file      | directory      | Write `directory/<input base name>.<ext>`           |
| directory | —              | Process every image, nothing written                |
| directory | directory      | Write every image as `<base name>.<ext>` (directory is created if missing) |
| directory | existing file  | Error                                               |

The output format is chosen as follows:
1. `-f` if given;
2. otherwise, when writing a single file, the extension of that file (`-o result.png` writes PNG);
3. otherwise PGM.

When writing to a directory, every output keeps its input base name with the extension of the output
format (`scan01.jpg` → `out/scan01.pgm`).

A directory is scanned non-recursively for files with a supported extension; hidden files and anything
else are ignored. The actual format of every file is detected from its contents, not its extension.

Before anything is written, the run is refused if:
- the output would overwrite the input (same file, or output directory equal to the input directory);
- two inputs would map to the same output name (e.g. `scan.png` and `scan.jpg` both becoming `scan.pgm`).

Images that fail at any stage are reported, the rest are still processed, and the exit status is non-zero.

Examples:
```bash
# read and write a whole directory on 8 threads
./bin/imgfilter -t 8 -o out/ data/

# JPEG X-rays in, PNG out, with progress bars and a benchmark file
./bin/imgfilter -p -b results.json -f png -o out/ xrays/

# benchmark reading only, JSON to stdout
./bin/imgfilter -b - data/ | jq .results.read
```

## Denoising

`-d` enables **Non-Local Means** denoising, following Buades, Coll and Morel, *Non-Local Means Denoising*,
Image Processing On Line 1 (2011). Every pixel is replaced by a weighted average of the pixels in a search
window around it, weighted by how similar the patches around them are:

```
d²(p,q) = mean squared difference of the patches around p and q
w(p,q)  = exp( −max(d² − 2σ², 0) / h² ),   h = k·σ
out(p)  = Σ w(p,q)·I(q) / Σ w(p,q)
```

Two noisy copies of the same patch differ by 2σ² on average, which is subtracted so that identical
structure gets full weight. The pixel itself is weighted like its most similar neighbor.

| Option   | Parameter                          | Default                  | Effect |
| -------- | ---------------------------------- | ------------------------ | ------ |
| `-P <r>` | patch radius, patches (2r+1)²      | 2 (5×5)                  | larger: more robust similarity under strong noise, less fine detail |
| `-S <r>` | search radius, window (2r+1)²      | 10 (21×21)               | larger: more candidates, stronger denoising; cost ∝ (2r+1)² |
| `-H <k>` | strength, h = k·σ                  | 0.4                      | larger: smoother; smaller: more noise kept |
| `-N <σ>` | noise standard deviation           | estimated per image      | scale of h and of the 2σ² offset |

The defaults are the paper's recommendation for moderate noise. When `-N` is not given, σ is estimated
for every image with Immerkær's method (*Fast Noise Variance Estimation*, 1996), using integer sums so the
estimate does not depend on the thread count. The estimate reads low on images with large saturated
areas (e.g. pure black backgrounds), where clipping has removed part of the noise; use `-N` there.

### Algorithm

Comparing every pair of patches directly costs (2S+1)²·(2P+1)² operations per pixel, 11,025 with the
defaults. Instead, for every offset of the search window, the squared differences between the image and
its shifted copy are accumulated into an integral image (summed-area table), from which any patch sum is
read with 4 lookups (Darbon et al., ISBI 2008). The cost becomes about (2S+1)² per pixel, independent of
the patch size. On top of that:

- **Weight table.** A weight depends only on the integer patch sum, so the weights of all possible sums
  are computed once per image; the table holds exactly the values `expf()` would return.
- **Offset blocking.** Offsets are processed in groups of 4: their integral images are built in one
  interleaved pass whose independent running sums the processor overlaps, and every pixel's accumulators
  are loaded and stored once per group instead of once per offset, with the additions in the same order.
- **Chunked rejection.** The patch sums of a row are computed in a branch-free, vectorizable loop, and
  chunks of 16 pixels in which every candidate is rejected (most of them on clean images) are skipped.
- **Mirrored padding.** The image is padded once so the inner loops need no bounds checks, and 32-bit
  integral values are allowed to wrap, since patch sums are exact modulo 2³².

Single-thread times on a 512×512 crop of a chest X-ray, defaults (P=2, S=10):

| Input                  | direct comparison | this implementation |
| ---------------------- | ----------------: | ------------------: |
| clean X-ray (σ ≈ 0.8)  | 2.10 s            | 0.15 s              |
| noise σ = 10 added     | 3.32 s            | 0.36 s              |
| noise σ = 40 added     | 3.47 s            | 0.34 s              |

During development the output was verified against a direct implementation and a double-precision
reference: it is identical to the direct method bit for bit, and to the reference except at rounding
ties (a result within float precision of x.5, one gray level), which only occur with degenerate settings
such as 1×1 patches.

Measured and rejected: compiling for the host CPU (`-march=native`) does not help, since the hot loops are
limited by the integral-image recurrence and by data-dependent branches, not by vector width; ignoring
weights below 10⁻³ as OpenCV does is 18–24% *slower* on noisy images, because rejecting about half the
candidates makes the branch unpredictable, and it changes the output; a branch-free accumulation loop is
faster on very noisy images but slower on clean ones. Candidate preselection and blockwise NLM (Coupé et
al., 2008) reduce the number of weight computations, which here are not the dominant cost. Symmetric
weights would save up to ~1.5× at the price of exact reproducibility.

### Parallelization

The work of every image is split into bands of rows. For each batch:

1. every image is prepared (noise estimate, padding, weight table): one image per thread when the batch
   has at least one image per thread, otherwise one image at a time with all threads;
2. the bands of *all* images of the batch form one pool that the threads draw from dynamically, bands of
   the widest images first, so the last bands to finish are the cheapest;
3. the outputs are collected.

A large image is thus spread over all threads, and the idle time at the end of a batch is at most about
one band. Band height is about four bands per thread over the batch, 16–32 rows (every band recomputes
2P extra rows of integral image, so bands should not be too small); with fewer images than threads,
each image gets a multiple of the thread count of bands, so a single image divides evenly.

The output is identical for any thread count; `-t 1` is the serial baseline.

```bash
# denoise a directory
./bin/imgfilter -d -o clean/ noisy/

# stronger smoothing with a known noise level
./bin/imgfilter -d -H 0.6 -N 20 -o clean.pgm noisy.png
```

## Edge Detection

`-e` runs Canny edge detection (J. Canny, *A Computational Approach to Edge Detection*, IEEE PAMI, 1986);
with `-d` as well, it runs on the denoised image. The output is the edge map: 255 on edges, 0 elsewhere.

| Step | | Option |
| ---- | --- | ------ |
| 1. Gaussian blur | suppresses noise before differentiating | `-G <σ>` (default 1.4; 0 = none) |
| 2. Sobel gradients | strength and direction of the intensity change | |
| 3. Non-maximum suppression | keeps a pixel only if it is the strongest along its gradient direction, thinning edges to one pixel | |
| 4. Hysteresis | pixels above the high threshold are edges; pixels above the low threshold are edges if 8-connected to one | `-l <low>` (default 20), `-u <high>` (default 50) |

Thresholds are gradient magnitudes of the 8-bit image (Sobel), the same units as OpenCV's Canny. On a chest
X-ray the defaults find the ribs and the larger vessels; the result agrees with OpenCV's Canny (same blur
and thresholds, L2 gradient) to within one pixel for 86–92% of the edge pixels, the difference coming
mostly from the blur, which here keeps 1/256 gray-level precision instead of rounding to 8 bits.

**Integer arithmetic.** The Gaussian weights are integers summing to 256, the blurred image is kept in
units of 1/256 gray level, gradient magnitudes are compared squared as 64-bit integers, and directions are
quantized by exact integer comparisons with tan 22.5°. The per-pixel arithmetic lives in `canny_core.h`,
compiled into both the C code and the CUDA kernels, so both run exactly the same operations and produce
identical edge maps. The CPU implementation was checked against an independent NumPy implementation of
the same specification.

**Parallelization.** On the CPU, as for denoising, the images of a batch are split into bands of rows
(16–64) that the threads draw from one pool, widest images first. Each band computes its blur and gradients
in small per-thread buffers, including a few border rows that neighboring bands compute again; no
full-image intermediate arrays are kept, so memory does not grow with the batch. The thread that completes
the last band of an image runs its hysteresis (a flood fill from the strong edges) right away, overlapping
it with other images' bands. Away from the image border the per-pixel functions skip the mirroring of
indices (same values, same order), which together with the cache-sized buffers made the single-thread
time 2.3× lower (199 → 87 ms on a 1857×1317 X-ray).

On the GPU, the per-pixel steps are one thread per pixel, and hysteresis is iterative:
each block propagates edges through weak pixels within its 32×32 tile in shared memory until nothing
changes (`__syncthreads_or`), and the host relaunches until a launch changes nothing, checking the
"changed" flag only every 4th launch to save synchronizations. Only *active* tiles do any work: all of
them in the first launch, then only those whose neighbor changed a pixel on their shared border (a side,
or a corner for diagonal neighbors); a tile can have work left only in that case, so the result stays
exact. On chest X-rays about 5–10% of the tile runs of all launches remain, and launches after
convergence cost almost nothing. The result is the
unique set of weak pixels connected to a strong one, independent of the propagation order, so it equals
the flood fill's. Buffers stay on the GPU between steps: one upload and one download per image. With
`-d -e`, the denoised image stays on the GPU for edge detection instead of being downloaded and uploaded
again.

## GPU Filters

`-g` runs the filters (denoising and/or edge detection) on the GPU. For denoising, the CPU still prepares
every image (noise estimate, mirrored padding, weight table) and the GPU computes the denoised pixels.

**Kernel.** One thread per output pixel, in blocks of 32×16. A block loads its tile of the padded image,
plus a border of P+S pixels, into shared memory once, then loops over all offsets of the search window:
for each offset it computes the squared differences between the tile and its shifted copy in shared
memory, sums them over the patch width, and every thread sums its column over the patch height. That is
the pixel's exact integer patch sum. Weights come from the table built by the CPU, and every thread keeps
its accumulators in registers for the whole loop, so they cost no memory traffic. (The integral-image
method used on the CPU relies on sequential prefix sums and on streaming the image once per offset,
which suits a GPU poorly; per-block shared-memory sums are the standard GPU formulation.)

**Streams.** Images go through the GPU in two alternating slots, each with its own CUDA stream, device
buffers and pinned (page-locked) host buffers. While one image's kernels run, the next image is uploaded
and the previous one downloaded in the other slot: copies from pinned memory are asynchronous and overlap
with the kernels, and each slot's stream orders the work on its buffers. Only the hysteresis loop waits
for the GPU, and only for its own image's stream; its first launch group (the most expensive, processing
every tile) is enqueued before the host copies the neighboring images to and from pinned memory, so the
GPU works through it meanwhile instead of waiting for the host. This matters most for edge detection alone, whose light
kernels would otherwise leave the copies (about 40% of its GPU time) exposed. With `-d -e`, the denoised
image stays on the GPU for edge detection.

**Exactness.** The GPU computes the same integer patch sums, uses the same weights, adds the offsets in
the same order, and is compiled without fused multiply-add (`-fmad=false`, and no fast math), so its
output is **bit-for-bit identical** to the CPU's. To verify on a machine with a GPU:

```bash
./bin/imgfilter -d    -o cpu/ data/
./bin/imgfilter -d -g -o gpu/ data/
diff -r cpu/ gpu/ && echo identical
```

An image whose parameters need a larger weight table than the limit (very strong noise with very large
patches) is denoised on the CPU instead, with the same result.

**Pipeline.** With the GPU, three batches are in flight: while the GPU filters batch *k*, the CPU reads,
decodes and prepares batch *k+1* and encodes and writes batch *k−1*. Two OpenMP sections run side by
side: the GPU one is a single thread that sleeps while waiting for the GPU (blocking synchronization), the
CPU one uses all threads in its own nested parallel regions. Memory is therefore about three batches.

On the CPU alone such a pipeline gains nothing: every stage already keeps all cores busy (measured: 0.3%
idle time over a full run). With the GPU, the CPU would otherwise sit idle during denoising, so the
pipeline hides nearly all of the I/O and codec work behind it.

With the pipeline, stage wall times overlap, so they add up to more than `pipeline_time`; per-image
denoise time is the CPU preparation plus the GPU run.

## Test Data with Artificial Noise

The chest X-ray datasets are clean, so they measure speed but not denoising quality. `tools/noise.py`
makes noisy copies of an image or a directory of images, and `tools/compare.py` measures the result
against the clean originals (PSNR). Both need Python with numpy and Pillow (Arch: `python-numpy
python-pillow`).

```bash
tools/noise.py data/clean data/noisy                   # Poisson noise, dose 10-50% per image
tools/noise.py --dose 0.2 data/clean data/noisy_20     # fixed dose
tools/noise.py --model gaussian --sigma 20 in.png out.png
./bin/imgfilter -d -o data/denoised data/noisy
tools/compare.py data/clean data/noisy data/denoised   # PSNR per set (--per-image: per image)
```

- **Models.** `poisson` (default): signal-dependent photon-counting noise; a pixel of normalized
  brightness v receives on average v · photons · dose photons, drawn from a Poisson distribution, so a
  lower dose means stronger noise, strongest in dark regions. It treats brightness as proportional to the
  photon count, which in a radiograph displayed with bone white is the other way round: a model of
  signal-dependent noise, not a physical low-dose simulation. `gaussian`: additive white noise of a given
  σ, the usual setting in the denoising literature.
- **Lossless output** (PNG, or PGM with `--format pgm`): saving noisy images as JPEG would partly remove
  the noise again and add compression artifacts.
- **Reproducible**: each image's noise is drawn from a generator seeded with `--seed` and the file name,
  so a file always gets the same noise, whatever the order or subset processed.
- **Manifest**: `noise.csv` records every image's parameters and the standard deviation of the noise
  actually added, the ground truth for evaluating denoising and the noise estimate.

Example, four 512×512 crops of a chest X-ray with Poisson noise (dose 10–50%), default NLM parameters:

| | PSNR (mean) | PSNR (worst image) |
| --- | ---: | ---: |
| noisy | 27.45 dB | 24.09 dB |
| denoised | **34.47 dB** | **33.62 dB** |

The noise level estimated by `imgfilter` (mean σ 11.32) matched the added noise (11.41) to within 2%.

## Batch Processing

Images are processed in batches of `-B` images. Each batch passes through every stage, each stage
parallel over the batch, and is then released:

```
for each batch of B images:
    read → decode → [filters] → encode → write      (each stage: parallel over the batch)
    free the batch
```

Within a batch, every buffer is also freed as soon as it is no longer needed: the file contents after
decoding, the pixels after encoding, the encoded data after writing.

Peak memory therefore scales with `B`. For 400 images of 1024×1024 (400 MiB of pixels):

| `-B` | Batches | Peak RSS |
| ---: | ------: | -------: |
| 0 (all) | 1 | 465 MiB |
| 128 | 4 | 183 MiB |
| 32 | 13 | 59 MiB |
| 8 | 50 | 22 MiB |

Output is identical for every batch size and thread count. Each stage ends with a synchronization point
per batch, where threads wait for the slowest image; with `B` much larger than the thread count this cost
is negligible.

With `-p`, a progress bar shows each run of the pipeline (and, when benchmarking, each warmup and
timed trial). Progress is weighted by cost: every I/O and codec stage counts one unit per image, and
denoising counts one unit per search offset (440 with the defaults), shared out over each batch in
proportion to the images' pixel counts and advanced band by band. The bar therefore moves at a
roughly constant rate even though denoising takes over 90% of the time; it cannot be exact, since
the cost of a band also depends on its content (flat noisy regions keep more candidates than edges).

## Image Formats

| Format | Read | Write | Implementation |
| ------ | :--: | :---: | -------------- |
| PGM (Netpbm `P5` 8/16-bit, `P2` ASCII) | ✓ | ✓ | built-in |
| PNG (8/16-bit, any color type)         | ✓ | ✓ | stb_image / stb_image_write |
| JPEG (baseline, progressive)           | ✓ |   | stb_image |
| BMP, TGA                               | ✓ |   | stb_image |

All images are converted to 8-bit grayscale on decode: color is converted to luma and 16-bit samples
are rescaled.

**PGM is the default output** because it is uncompressed: encoding is a header plus a copy of the pixels,
while PNG encoding runs deflate compression and is by far the most expensive stage of the pipeline.
Use `-f png` when the smaller files are worth it.

Scientific formats (HDF5, DICOM, NumPy `.npy`) are not read directly; convert them to PGM (16-bit is
supported, so no dynamic range is lost) or PNG first.

## Benchmarking

### Single configuration

With `-b`, the pipeline runs `-w` warmup trials (not recorded), then `-n` timed trials, and a JSON
document is written containing:
- `sys_info` — CPU model, logical cores, RAM and swap, and the GPU used (name, compute capability,
  multiprocessors, memory, driver and runtime versions) or `null`
- `benchmark_info` — timestamp, threads, trials, warmup trials, batch size and number of batches
- `dataset_info` — input/output paths, output format, number of images per input format, total pixels,
  range of dimensions
- `denoise` — `null`, or the device (`cpu` or `cuda`), the NLM parameters and the noise levels σ used
  (mean / min / max over the images)
- `edges` — `null`, or the device and the Canny parameters (with the Gaussian radius used)
- `results` — for each stage: images processed and failed, data volume, throughput, and two timing
  summaries (mean / median / standard deviation / min / max / total): `wall_time`, the stage's elapsed
  time over the trials, and `per_image`, the individual image times over all trials; `null` for stages
  that did not run
- `pipeline_time` — elapsed time of the whole pipeline over the trials
- `gpu_time` — with `-g`: the GPU's own time (CUDA events recorded right around each piece of work, so
  idle time between them is not counted), per trial, for uploading images and weight tables, the
  denoising kernel, the edge kernels (of which hysteresis), and downloading results, plus the
  number of hysteresis launches, the hysteresis tiles processed, and how many all launches would have
  processed with every tile active, totals per trial with statistics over trials;
  `null` otherwise
- `memory` — peak resident memory, and major/minor page faults during the timed trials

| Stage    | Work                       | Data volume (`data_mib`) |
| -------- | -------------------------- | ------------------------ |
| `read`   | file → memory (pure I/O)   | bytes read               |
| `decode` | memory → pixels (CPU)      | pixel bytes produced     |
| `denoise`| pixels → pixels (CPU/GPU)  | pixel bytes filtered     |
| `edges`  | pixels → edge map (CPU/GPU)| pixel bytes filtered     |
| `encode` | pixels → memory (CPU)      | pixel bytes consumed     |
| `write`  | memory → file (pure I/O)   | bytes written            |

Throughput uses the median wall time. `per_image.total_time_s` divided by the total wall time of the
trials shows how many threads were busy on average; it is not a speedup, which needs a single-thread
run for reference.

A non-zero `major_page_faults` means pages had to be read back from disk (e.g. swap) during the timed
trials, and the timings are not representative.

Reading is dominated by the page cache: a dataset read once is served from memory afterwards. State
the cache condition of every measurement; use `-w 1` for warm-cache results, or the cold mode of the
sweep script below.

### Sweeps

`tools/bench.sh` runs one `imgfilter` process per configuration and collects the JSON files, with a
record of the environment, in a results directory:

```bash
# threads 1..nproc, default batch size, warm cache, no output
tools/bench.sh data/xrays

# threads x batch sizes, writing PGM
tools/bench.sh -t "1 2 4 8 16" -B "16 64 256 1024" -f pgm data/xrays

# cold cache: page cache dropped (sudo) before every run
tools/bench.sh -c cold -n 3 -t "8 16" data/xrays

# denoising over thread counts, with filter parameters
tools/bench.sh -d -t "1 2 4 8 16" -n 3 -x "-S 7" data/xrays_subset

# one image, speedup over one thread
tools/bench.sh -d -t "1 2 4 8 16" -n 3 image.jpeg

# GPU
tools/bench.sh -d -x "-g" -t 16 -n 3 data/xrays_subset
```

When thread count 1 is part of a sweep, the summary shows each run's speedup over the 1-thread run of
the same configuration (denoise stage time if it ran, otherwise pipeline time).

Threads are bound to physical cores first (`OMP_PROC_BIND=close`, `OMP_PLACES=cores`); beyond one
thread per core they share cores through hyperthreading. Images are written to a scratch directory
inside the results directory, never to `/tmp`, which is often RAM-backed. Run `tools/bench.sh -h` for
all options.

## Project Structure

```
src/
├── main.c          Entry point: argument handling and pipeline stages
├── args.[ch]       Command-line parsing
├── io.[ch]         Input/output resolution and parallel loading/saving
├── image.[ch]      Image container and codecs
├── nlm.[ch]        Non-Local Means denoising and noise estimation
├── canny.[ch]      Canny edge detection (CPU)
├── canny_core.h    Canny per-pixel arithmetic, shared by CPU and GPU
├── gpu.h           GPU backend interface
├── gpu.cu          CUDA kernel and wrappers (the only CUDA file)
├── gpu_none.c      GPU backend for builds without CUDA
├── progress.[ch]   Thread-safe progress bar
├── benchmark.[ch]  Timing, statistics and system information
├── json.[ch]       JSON output
├── error.[ch]      Error reporting
└── external/       Vendored third-party code
    ├── stb.c           Compilation unit for the stb libraries
    ├── stb_image.h
    └── stb_image_write.h
tools/
├── bench.sh        Benchmark sweeps over threads and batch sizes
├── noise.py        Noisy test images (Poisson or Gaussian), with a manifest
└── compare.py      PSNR of processed images against clean references
```

## Third-Party Code

PNG, JPEG, BMP and TGA support uses [stb_image and stb_image_write](https://github.com/nothings/stb)
by Sean Barrett (public domain / MIT), vendored in `src/external/`. They are compiled into the program,
so building needs nothing beyond a C compiler with OpenMP. stb_image keeps its error state in
thread-local storage, which allows decoding from concurrent OpenMP threads.

---

<p align="center"><sub>September 2026 • Aristotle University of Thessaloniki</sub></p>
