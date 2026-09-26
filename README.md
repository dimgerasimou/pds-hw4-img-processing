# Parallel Image Filters — CUDA and OpenMP Implementations

<p align="center">
  <img src="https://img.shields.io/badge/parallelism-OpenMP-blue" alt="OpenMP">
  <img src="https://img.shields.io/badge/acceleration-CUDA-green" alt="CUDA">
</p>

Assignment #4 of the **Parallel and Distributed Systems** coursework: [parallel-distributed-systems](https://github.com/dimgerasimou/parallel-distributed-systems)

**`imgfilter`** is a batch **image processing tool** implementing **Non-Local Means (NLM) denoising** and
**Canny edge detection** on multicore CPUs (**OpenMP**) and **single NVIDIA GPUs** (**CUDA**), with a strong
emphasis on comparing parallelization strategies and reproducible benchmarking.

> **Status:** the I/O pipeline, benchmarking and progress reporting are implemented.
> The filters (NLM, Canny) and the CUDA backend are in progress.

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
- Configurable thread count
- Thread-safe **progress bars** for every stage
- Automated benchmarking with **JSON output**: system information, dataset information, and per-stage
  wall time, throughput and per-image statistics
- Planned:
  - **NLM denoising** — sequential, OpenMP and CUDA
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
| `-b <file>`   | Write benchmark results as JSON (`-` for stdout)        |
| `-p`          | Show progress bars (on stderr)                          |
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

With `-b`, a JSON document is written containing:
- `sys_info` — CPU model, logical cores, RAM and swap
- `benchmark_info` — timestamp and thread count
- `dataset_info` — input/output paths, output format, number of images per input format, total pixels,
  range of dimensions
- `results` — for each stage: images processed and failed, data volume, wall time, throughput, and
  per-image mean / median / standard deviation / min / max; `null` for stages that did not run
- `cpu_peak_rss_gb` — peak memory usage of the process

| Stage    | Work                       | Data volume (`data_mib`) |
| -------- | -------------------------- | ------------------------ |
| `read`   | file → memory (pure I/O)   | bytes read               |
| `decode` | memory → pixels (CPU)      | pixel bytes produced     |
| `encode` | pixels → memory (CPU)      | pixel bytes consumed     |
| `write`  | memory → file (pure I/O)   | bytes written            |

For a stage, `wall_time_s` is the elapsed time of the whole parallel stage, while
`per_image.total_time_s` is the sum of the individual image times; their ratio is the effective
parallelism achieved.

## Project Structure

```
src/
├── main.c          Entry point: argument handling and pipeline stages
├── args.[ch]       Command-line parsing
├── io.[ch]         Input/output resolution and parallel loading/saving
├── image.[ch]      Image container and PGM reading/writing
├── progress.[ch]   Thread-safe progress bar
├── benchmark.[ch]  Timing, statistics and system information
├── json.[ch]       JSON output
├── error.[ch]      Error reporting
└── external/       Vendored third-party code
    ├── stb.c           Compilation unit for the stb libraries
    ├── stb_image.h
    └── stb_image_write.h
```

## Third-Party Code

PNG, JPEG, BMP and TGA support uses [stb_image and stb_image_write](https://github.com/nothings/stb)
by Sean Barrett (public domain / MIT), vendored in `src/external/`. They are compiled into the program,
so building needs nothing beyond a C compiler with OpenMP. stb_image keeps its error state in
thread-local storage, which allows decoding from concurrent OpenMP threads.

---

<p align="center"><sub>September 2026 • Aristotle University of Thessaloniki</sub></p>
