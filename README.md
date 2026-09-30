# Batch Image Denoise — OpenMP and CUDA Implementation

<p align="center">
  <img src="https://img.shields.io/badge/parallelism-OpenMP%20%2B%20CUDA-blue" alt="OpenMP + CUDA">
</p>

Assignment #4 of the **Parallel and Distributed Systems** coursework: [parallel-distributed-systems](https://github.com/dimgerasimou/parallel-distributed-systems)

A **batch image denoising and edge detection** tool for grayscale images, using **Non-Local Means** and
**Canny**, on **multi-core CPUs (OpenMP)** and **NVIDIA GPUs (CUDA)**.

## Overview

The goal of this project is to process large sets of noisy X-ray and CT images quickly, and to measure how well it does.

It includes:
- Non-Local Means denoising with automatic noise estimation
- Canny edge detection, optionally after denoising
- A CPU pipeline (OpenMP) and a GPU pipeline (CUDA) with bit-identical results
- A benchmark script for speed and quality on real data

## Features

- **Non-Local Means** denoising with a patch and a search window (default 5×5 and 21×21)
- **Automatic noise level and strength** per image, also for the correlated noise of CT reconstructions
- **Canny** edge detection with integer arithmetic, so that CPU and GPU agree exactly
- **Batch pipeline**: images are read, decoded, filtered, encoded and written in batches. On the GPU the CPU prepares other batches while the kernels run
- Input: PGM, PNG, JPEG, BMP, TGA (converted to grayscale). Output: PGM, PNG
- Benchmark mode with warmup and trials, and machine-readable **JSON output**
- Quality evaluation against a ground truth (PSNR and edge F1)

## Build

### Requirements
- C compiler (`gcc` or `clang`) with OpenMP
- `make`
- CUDA Toolkit (optional, for the GPU)
- Python 3 with `numpy` and `Pillow` (for the tools)

### Compile
```bash
make
```

Produces `bin/imgfilter`.

## Usage

```bash
./bin/imgfilter [OPTIONS] <input>
```

The input is an image or a directory of images.

```bash
./bin/imgfilter -d -o clean/ noisy/              # denoise a directory
./bin/imgfilter -d -g -p -o clean/ noisy/        # on the GPU, with progress bars
./bin/imgfilter -d -e -g -o edges/ noisy/        # edges of the denoised images
./bin/imgfilter -e -o edges.png image.png        # edges only
```

| Option | Meaning |
|---|---|
| `-d` | Denoise (`-P` patch radius, `-S` search radius, `-H` strength, `-N` noise level) |
| `-e` | Edge detection (`-G` blur, `-l` low and `-u` high threshold) |
| `-g` | Run the filters on the GPU |
| `-t`, `-B` | Threads, and images per batch |
| `-b`, `-n`, `-w` | Benchmark JSON file, trials, warmup trials |
| `-o`, `-f`, `-p` | Output file or directory, output format, progress bars |

Run with `-h` to see all options.

## Benchmarking

`tools/bench.py` runs the speed benchmarks and the quality evaluation, and writes everything to `results/<timestamp>/`. Its steps can be skipped with `--skip`:

| Step | What it does |
|---|---|
| `batch` | batch-size sweep on the CPU and the GPU |
| `threads` | thread sweep on the CPU |
| `quality` | PSNR and edge F1 against the ground truth, and a check that CPU and GPU outputs are identical |

The ground truth is either a directory of clean images (`--reference`, files match by name), or the input images themselves with noise added (`--noise poisson` or `gaussian`).

```bash
# quality on every image of a dataset with a ground truth
tools/bench.py --reference data/aapm/reference --skip batch,threads data/aapm/input

# speed and quality on a random sample of 500 images
tools/bench.py --reference data/2detect/reference --sample 500 data/2detect/input

# images without a ground truth: noise is added to the originals
tools/bench.py --sample 500 data/chest-xray/reference
```

Useful options are `--devices cpu,gpu`, `--batches`, `--threads`, `-n` and `-w` for the trials, `--nlm "-S 7 -H 1.0"` to pass options to the denoiser, and `--keep-outputs` to keep the processed images.

The results directory holds `summary.md` (tables), `runs.csv` and `quality.csv` (per image), `environment.txt` and one JSON file per run.

### Datasets

`tools/convert.py` turns CT data (`.npy` or `.tif`) into the 8-bit PNG pairs the tools expect, with one fixed window for both images of a pair. `tools/noise.py` writes noisy copies of a directory of images.

In the folder `scripts`, you can run from the project directory shell scripts that automatically download and convert the datasets for you. They need `wget` and `unzip` (and `7z` for AAPM).

## Performance Results

Detailed benchmark results and tables are available in [docs/performance.md](docs/performance.md).

## Notes

- The automatic mode is a compromise: tuning `-H` or `-N` for one dataset can gain a few tenths of a dB
- Strong smoothing raises the PSNR but blurs fine edges, so `-d -e` uses a milder strength
- Structured artifacts (such as streaks in photon-starved CT slices) are not noise and are not removed
- Pairs of images must be registered and of the same size for the quality step

---

<p align="center"><sub>September 2026 • Aristotle University of Thessaloniki</sub></p>
