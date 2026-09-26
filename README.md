# Parallel Image Filters — CUDA and OpenMP Implementations

<p align="center">
  <img src="https://img.shields.io/badge/parallelism-OpenMP-blue" alt="OpenMP">
  <img src="https://img.shields.io/badge/acceleration-CUDA-green" alt="CUDA">
</p>

Assignment #4 of the **Parallel and Distributed Systems** coursework: [parallel-distributed-systems](https://github.com/dimgerasimou/parallel-distributed-systems)

**`imgfilter`** is a batch **image processing tool** implementing **Non-Local Means (NLM) denoising** and
**Canny edge detection** on multicore CPUs (**OpenMP**) and **single NVIDIA GPUs** (**CUDA**), with a strong
emphasis on comparing parallelization strategies and reproducible benchmarking.

> **Status:** the I/O pipeline, benchmarking, and NLM denoising with OpenMP are implemented.
> Canny edge detection and the CUDA backend are in progress.

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
- **Non-Local Means denoising** with OpenMP, parallelized either across images or within each image
- Planned:
  - **NLM denoising** — CUDA
  - **Canny edge detection** — sequential, OpenMP and CUDA

## Build

### Requirements
- C compiler with OpenMP support (`gcc` or `clang`)
- `make`

### Compile
```bash
make
```

Produces:
```
bin/imgfilter
```

Run `make help` for all targets and overrides.

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

| Option      | Parameter                          | Default                  | Effect |
| ----------- | ---------------------------------- | ------------------------ | ------ |
| `-P <r>`    | patch radius, patches (2r+1)²      | 2 (5×5)                  | larger: more robust similarity under strong noise, less fine detail |
| `-S <r>`    | search radius, window (2r+1)²      | 10 (21×21)               | larger: more candidates, stronger denoising |
| `-H <k>`    | strength, h = k·σ                  | 0.4                      | larger: smoother; smaller: more noise kept |
| `-N <σ>`    | noise standard deviation           | estimated per image      | scale of h and of the 2σ² offset |
| `-A <name>` | method: `integral` or `direct`     | `integral`               | performance only; the output is identical |
| `-m <mode>` | parallelization (see below)        | `auto`                   | performance only; the output is identical |

The defaults are the paper's recommendation for moderate noise. When `-N` is not given, σ is estimated
for every image with Immerkær's method (*Fast Noise Variance Estimation*, 1996), using integer sums so the
estimate does not depend on the thread count. The estimate reads low on images with large saturated
areas (e.g. pure black backgrounds), where clipping has removed part of the noise; use `-N` there.

**Methods** (`-A`). Both produce bit-for-bit identical output:

- `direct` compares every pair of patches pixel by pixel: (2S+1)²·(2P+1)² operations per pixel
  (11,025 with the defaults). A comparison stops as soon as its partial sum shows the weight will be
  negligible, which skips most of the work on images with little noise.
- `integral` (default) exchanges the loops: for every offset of the search window, the squared
  differences between the image and its shifted copy are accumulated into an integral image
  (summed-area table), from which any patch sum is read with 4 lookups. The cost is about (2S+1)² per
  pixel, independent of the patch size (Darbon et al., ISBI 2008). The image is processed in bands of
  about 32 rows so that each band's integral image and accumulators stay in the L2 cache; 32-bit
  integral values are allowed to wrap, since patch sums are exact modulo 2³².

In both, a weight depends only on the integer patch sum, so the weights of all possible sums are
computed once per image into a table instead of calling `expf()` for every candidate; the table holds
exactly the values `expf()` would return.

Single-thread times, 512×512 crop of a chest X-ray, defaults (P=2, S=10):

| Input                  | direct (original) | direct | integral |
| ---------------------- | ----------------: | -----: | -------: |
| clean X-ray (σ ≈ 0.8)  | 2.10 s            | 0.98 s | 0.17 s   |
| noise σ = 10 added     | 3.32 s            | 2.08 s | 0.34 s   |
| noise σ = 40 added     | 3.47 s            | 2.31 s | 0.42 s   |

The integral method is 5–12× faster than the optimized direct method and 10–19× faster than the
original. On clean images, rejected candidates are skipped with a well-predicted branch; a branch-free
loop would be faster on very noisy images but slower on clean ones.

**Parallelization** (`-m`):

| Mode    | Work distribution                                                         |
| ------- | ------------------------------------------------------------------------- |
| `image` | threads take whole images of the batch; each image is filtered serially  |
| `pixel` | images are filtered one at a time; the threads split each image's rows   |
| `auto`  | `image` when the batch holds at least one image per thread, else `pixel` |

`pixel` is the only mode that speeds up a single image (rows for `direct`, bands for `integral`);
`image` avoids synchronization inside the filter but depends on having enough images, of similar size,
per batch. The output is identical for
every mode and thread count; `-t 1` is the serial baseline.

```bash
# denoise a directory
./bin/imgfilter -d -o clean/ noisy/

# stronger smoothing, known noise level, one image on all threads
./bin/imgfilter -d -H 0.6 -N 20 -m pixel -o clean.pgm noisy.png
```

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
timed trial).

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
- `sys_info` — CPU model, logical cores, RAM and swap
- `benchmark_info` — timestamp, threads, trials, warmup trials, batch size and number of batches
- `dataset_info` — input/output paths, output format, number of images per input format, total pixels,
  range of dimensions
- `denoise` — `null`, or the NLM parameters, parallelization mode, and the noise levels σ used
  (mean / min / max over the images)
- `results` — for each stage: images processed and failed, data volume, throughput, and two timing
  summaries (mean / median / standard deviation / min / max / total): `wall_time`, the stage's elapsed
  time over the trials, and `per_image`, the individual image times over all trials; `null` for stages
  that did not run
- `pipeline_time` — elapsed time of the whole pipeline over the trials
- `memory` — peak resident memory, and major/minor page faults during the timed trials

| Stage    | Work                       | Data volume (`data_mib`) |
| -------- | -------------------------- | ------------------------ |
| `read`   | file → memory (pure I/O)   | bytes read               |
| `decode` | memory → pixels (CPU)      | pixel bytes produced     |
| `denoise`| pixels → pixels (CPU)      | pixel bytes filtered     |
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

# denoising: both parallelization modes over thread counts, with filter parameters
tools/bench.sh -d -m "image pixel" -t "1 2 4 8 16" -x "-P 2 -S 7" data/xrays_subset

# denoising: both methods on one image, speedup over one thread
tools/bench.sh -d -a "direct integral" -m pixel -t "1 2 4 8 16" image.jpeg
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
├── progress.[ch]   Thread-safe progress bar
├── benchmark.[ch]  Timing, statistics and system information
├── json.[ch]       JSON output
├── error.[ch]      Error reporting
└── external/       Vendored third-party code
    ├── stb.c           Compilation unit for the stb libraries
    ├── stb_image.h
    └── stb_image_write.h
tools/
└── bench.sh        Benchmark sweeps over threads, batch sizes and modes
```

## Third-Party Code

PNG, JPEG, BMP and TGA support uses [stb_image and stb_image_write](https://github.com/nothings/stb)
by Sean Barrett (public domain / MIT), vendored in `src/external/`. They are compiled into the program,
so building needs nothing beyond a C compiler with OpenMP. stb_image keeps its error state in
thread-local storage, which allows decoding from concurrent OpenMP threads.

---

<p align="center"><sub>September 2026 • Aristotle University of Thessaloniki</sub></p>
