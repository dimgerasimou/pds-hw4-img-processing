# Performance

This report summarizes the benchmark runs produced by the current implementation: the speed of the CPU and GPU pipelines on a sample of each dataset, and the quality of the denoising and of the edge detection on every image of each dataset.

## Benchmark Environment

- **CPU**: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz (8 cores, 16 threads)
- **RAM**: 15.3 GiB
- **GPU**: NVIDIA GeForce RTX 3060 Laptop GPU (Compute Capability 8.6, 30 SMs, 5.7 GiB)
- **CUDA driver / runtime**: 13040 / 13040
- **Governor**: performance; threads bound with `OMP_PROC_BIND=close`, `OMP_PLACES=cores`
- **Commit**: 5be7a85

## Method

- **Speed**: a random sample of 500 images of each dataset (seed 0). A batch-size sweep on the CPU (all threads) and the GPU, and a thread sweep on the CPU (batch of 64), for denoising and for edge detection. Each configuration runs 3 timed trials after 1 warmup trial, and the median is reported. The pipeline time covers reading, decoding and filtering, without writing any output.
- **Quality**: every image of each dataset. PSNR of the noisy and of the denoised image against the ground truth, and the edge F1 (Canny edges of the noisy, respectively denoised, image against the edges of the ground truth, with a tolerance of 1 pixel). The CPU and GPU outputs are compared byte for byte.
- All runs use the default options: the noise level and the filter strength are estimated for every image.

## Datasets

| Dataset                 | Images |           Size (px) | Total (Mpx) | Ground truth / noise                                                                      |
| ----------------------- | -----: | ------------------: | ----------: | ----------------------------------------------------------------------------------------- |
| Chest X-ray (pneumonia) |  5,856 | 384–2916 × 127–2713 |       8,309 | the original images; Poisson noise is added (random dose of 10–50% per image, seed 0)     |
| Chest X-ray (mixed)     | 25,553 | 144–2916 × 124–2863 |      18,486 | the original images; Poisson noise is added (random dose of 10–50% per image, seed 0)     |
| AAPM Low-Dose CT        |  4,260 |             512×512 |       1,117 | full-dose image; the quarter-dose image is the input (noise simulated on the projections) |
| 2DeteCT                 |  1,000 |           1024×1024 |       1,049 | mode 2 image; the mode 1 image is the input (real noise)                                  |

- **Chest X-ray (pneumonia)**: The 5,856 pediatric chest radiographs of the Kaggle *Chest X-Ray Images (Pneumonia)* dataset (Kermany et al., 2018).
- **Chest X-ray (mixed)**: A collection of 25,553 chest radiographs of mixed origin and size: 10,641 tuberculosis, 9,088 normal and 5,824 pneumonia cases (by file name).
- **AAPM Low-Dose CT**: CT slices of the AAPM Low-Dose CT Grand Challenge (Mayo Clinic), from a community `.npy` mirror, converted to 8-bit PNG with one fixed window (`tools/convert.py`).
- **2DeteCT**: Lab CT slices of 2DeteCT (Kiss et al., 2023): the reconstructions of mode 1 (low dose) and mode 2 (high fidelity) of the same slice, converted to 8-bit PNG with one fixed window (`tools/convert.py`).

## Quality

Every image of each dataset.

| Dataset                 | Images | PSNR noisy (dB) | PSNR denoised (dB) | PSNR gain (dB) | Worst image (dB) | Edge F1 noisy | Edge F1 denoised | F1 gain | Noise σ used | Strength |
| ----------------------- | -----: | --------------: | -----------------: | -------------: | ---------------: | ------------: | ---------------: | ------: | -----------: | -------: |
| Chest X-ray (pneumonia) |  5,856 |           27.63 |              36.87 |          +9.24 |    21.94 → 28.30 |         0.691 |            0.800 |  +0.110 |         10.4 |     0.70 |
| Chest X-ray (mixed)     | 25,553 |           27.51 |              36.24 |          +8.74 |    21.34 → 26.34 |         0.738 |            0.798 |  +0.061 |         10.7 |     0.71 |
| AAPM Low-Dose CT        |  4,260 |           26.08 |              30.57 |          +4.48 |    18.16 → 18.39 |         0.821 |            0.852 |  +0.031 |         12.3 |     0.86 |
| 2DeteCT                 |  1,000 |           22.77 |              28.46 |          +5.69 |    14.57 → 15.32 |         0.249 |            0.749 |  +0.500 |         18.6 |     1.60 |

Worst image: the lowest PSNR among the noisy, and among the denoised, images. σ and strength are the mean over the images; the edge runs use a milder strength for near-white noise.

### Chest X-ray (pneumonia)

| Noisy PSNR (dB) | Images | PSNR gain (dB) | F1 noisy | F1 denoised | F1 gain | F1 lower than noisy |
| --------------- | -----: | -------------: | -------: | ----------: | ------: | ------------------: |
| 21.9 – 26.2     |  1,464 |         +10.59 |    0.499 |       0.755 |  +0.256 |              2 (0%) |
| 26.2 – 27.9     |  1,464 |          +9.55 |    0.697 |       0.796 |  +0.098 |             19 (1%) |
| 27.9 – 29.2     |  1,464 |          +8.83 |    0.764 |       0.818 |  +0.053 |            104 (7%) |
| 29.2 – 32.3     |  1,464 |          +7.98 |    0.803 |       0.833 |  +0.030 |           216 (15%) |

By noise level, noisiest quarter first. PSNR gain per image: mean +9.24 dB, median +9.21, smallest +2.95, largest +15.36; images with a lower PSNR after denoising: 0 of 5,856. Noise added: 10.89, used by imgfilter: 10.42. The CPU and GPU outputs are byte-identical, for denoising and for denoising with edges: **yes**.

By group, from the file names:

| Group     | Images | PSNR noisy (dB) | PSNR denoised (dB) | PSNR gain (dB) | F1 noisy | F1 denoised | F1 gain |
| --------- | -----: | --------------: | -----------------: | -------------: | -------: | ----------: | ------: |
| pneumonia |  4,273 |           27.63 |              37.15 |          +9.51 |    0.712 |       0.808 |  +0.096 |
| normal    |  1,583 |           27.63 |              36.12 |          +8.49 |    0.633 |       0.780 |  +0.147 |

**Observation**:

Poisson noise of about 11 gray levels is estimated 4% low (10.4 used, 10.9 added). Denoising gains 9.2 dB on average and never makes an image worse. The edge F1 rises from 0.69 to 0.80; the gain is largest on the noisiest images (+0.256) and smallest on the cleanest (+0.030), where 15% of the images end up with a lower edge score than the noisy input. The normal radiographs, whose noisy edges are the least reliable (F1 0.633), gain the most (+0.147).

### Chest X-ray (mixed)

| Noisy PSNR (dB) | Images | PSNR gain (dB) | F1 noisy | F1 denoised | F1 gain | F1 lower than noisy |
| --------------- | -----: | -------------: | -------: | ----------: | ------: | ------------------: |
| 21.3 – 26.1     |  6,389 |         +10.00 |    0.595 |       0.748 |  +0.153 |            327 (5%) |
| 26.1 – 27.8     |  6,388 |          +9.02 |    0.742 |       0.794 |  +0.052 |         1,164 (18%) |
| 27.8 – 29.1     |  6,388 |          +8.38 |    0.793 |       0.817 |  +0.024 |         1,913 (30%) |
| 29.1 – 38.5     |  6,388 |          +7.54 |    0.821 |       0.835 |  +0.014 |         2,233 (35%) |

By noise level, noisiest quarter first. PSNR gain per image: mean +8.74 dB, median +8.75, smallest -1.59, largest +15.97; images with a lower PSNR after denoising: 4 of 25,553. Noise added: 11.04, used by imgfilter: 10.70. The CPU and GPU outputs are byte-identical, for denoising and for denoising with edges: **yes**.

By group, from the file names:

| Group        | Images | PSNR noisy (dB) | PSNR denoised (dB) | PSNR gain (dB) | F1 noisy | F1 denoised | F1 gain |
| ------------ | -----: | --------------: | -----------------: | -------------: | -------: | ----------: | ------: |
| tuberculosis | 10,641 |           27.40 |              35.63 |          +8.23 |    0.781 |       0.810 |  +0.029 |
| normal       |  9,088 |           27.51 |              36.58 |          +9.07 |    0.714 |       0.783 |  +0.069 |
| pneumonia    |  5,824 |           27.70 |              36.83 |          +9.14 |    0.695 |       0.801 |  +0.106 |

**Observation**:

The results match the pneumonia set: +8.7 dB, and an edge F1 of 0.80 (from 0.74). Four of the 25,553 images lose PSNR (0.3 to 1.6 dB); all four are tuberculosis radiographs, the group with the smallest gains (+8.2 dB, F1 +0.029) and the most reliable noisy edges (F1 0.781). In the cleanest quarter, 35% of the images have a lower edge F1 after denoising.

### AAPM Low-Dose CT

| Noisy PSNR (dB) | Images | PSNR gain (dB) | F1 noisy | F1 denoised | F1 gain | F1 lower than noisy |
| --------------- | -----: | -------------: | -------: | ----------: | ------: | ------------------: |
| 18.2 – 24.4     |  1,065 |          +4.77 |    0.742 |       0.813 |  +0.071 |             52 (5%) |
| 24.4 – 25.8     |  1,065 |          +4.43 |    0.782 |       0.839 |  +0.057 |           115 (11%) |
| 25.8 – 27.7     |  1,065 |          +4.50 |    0.860 |       0.871 |  +0.011 |           478 (45%) |
| 27.7 – 34.9     |  1,065 |          +4.24 |    0.900 |       0.886 |  -0.014 |           704 (66%) |

By noise level, noisiest quarter first. PSNR gain per image: mean +4.48 dB, median +4.61, smallest +0.24, largest +6.19; images with a lower PSNR after denoising: 0 of 4,260. The CPU and GPU outputs are byte-identical, for denoising and for denoising with edges: **yes**.

**Observation**:

Denoising raises the PSNR by 4.5 dB on average, and no image gets worse. The edge F1 gain is concentrated in the noisier slices (+0.071 in the noisiest quarter); in the cleanest quarter Canny already copes with the noisy image and the edge score drops slightly (−0.014). The worst slice (18.2 dB) gains only 0.2 dB: slices dominated by structured artifacts rather than noise cannot be repaired by a denoiser.

### 2DeteCT

| Noisy PSNR (dB) | Images | PSNR gain (dB) | F1 noisy | F1 denoised | F1 gain | F1 lower than noisy |
| --------------- | -----: | -------------: | -------: | ----------: | ------: | ------------------: |
| 14.6 – 21.4     |    250 |          +4.08 |    0.252 |       0.628 |  +0.376 |              0 (0%) |
| 21.4 – 22.4     |    250 |          +4.82 |    0.248 |       0.757 |  +0.508 |              0 (0%) |
| 22.4 – 24.5     |    250 |          +5.56 |    0.238 |       0.793 |  +0.555 |              0 (0%) |
| 24.6 – 27.2     |    250 |          +8.29 |    0.259 |       0.819 |  +0.560 |              0 (0%) |

By noise level, noisiest quarter first. PSNR gain per image: mean +5.69 dB, median +5.13, smallest +0.75, largest +9.46; images with a lower PSNR after denoising: 0 of 1,000. The CPU and GPU outputs are byte-identical, for denoising and for denoising with edges: **yes**.

**Observation**:

Edge detection on the raw slices is almost useless (F1 0.25). After denoising it reaches 0.75, and every one of the 1,000 slices improves in both PSNR and edge F1. The real noise is spatially correlated and is estimated at 18.6 gray levels, with the strength at its upper end (1.6). The PSNR gain is largest on the cleanest slices (+8.3 dB) and smallest on the noisiest (+4.1 dB).

### Quality on the 500-image samples

| Dataset                 | Images | PSNR noisy (dB) | PSNR denoised (dB) | Edge F1 noisy | Edge F1 denoised |
| ----------------------- | -----: | --------------: | -----------------: | ------------: | ---------------: |
| Chest X-ray (pneumonia) |    500 |           27.49 |              36.91 |         0.689 |            0.798 |
| Chest X-ray (mixed)     |    500 |           27.47 |              36.30 |         0.742 |            0.798 |
| AAPM Low-Dose CT        |    500 |           26.20 |              30.70 |         0.829 |            0.855 |
| 2DeteCT                 |    500 |           22.80 |              28.48 |         0.247 |            0.751 |

## Speed

Times are medians over 3 trials of a 500-image sample. Throughput is in megapixels per second.

### Chest X-ray (pneumonia)

500 images, 710 Mpx.

#### Denoising

Batch sweep (CPU with all threads, GPU). Memory is the peak resident memory of the process:

| Batch | CPU time (s) | CPU memory (GB) | GPU time (s) | GPU memory (GB) | GPU speedup |
| ----- | -----------: | --------------: | -----------: | --------------: | ----------: |
| 4     |      104.866 |            0.25 |        5.079 |            0.42 |      20.65× |
| 16    |      101.718 |            0.28 |        4.975 |            0.66 |      20.45× |
| 64    |      100.553 |            0.55 |        5.118 |            1.20 |      19.65× |
| 256   |      100.497 |            1.45 |        5.455 |            2.38 |      18.42× |

Thread sweep (CPU, batch of 64):

| Threads | Time (s) | Speedup | Efficiency | Throughput (Mpx/s) |
| ------- | -------: | ------: | ---------: | -----------------: |
| 1       |  636.787 |   1.00× |       100% |                1.1 |
| 2       |  319.423 |   1.99× |       100% |                2.2 |
| 4       |  174.960 |   3.64× |        91% |                4.1 |
| 8       |  100.891 |   6.31× |        79% |                7.0 |
| 16      |  100.197 |   6.36× |        40% |                7.1 |

- **Best CPU**: 100.197 s (16 threads, batch 64), 7.1 Mpx/s
- **Best GPU**: 4.975 s (batch 16), 142.8 Mpx/s
- **GPU speedup**: 20.14× over the best CPU run, 128.00× over one thread

Stage times (median, s). The stages of different batches overlap, and on the GPU two images are in flight, so a stage can take longer than the pipeline:

| Stage    | CPU (16 threads, batch 64) | GPU (batch 16) |
| -------- | -------------------------: | -------------: |
| Read     |                      0.034 |          0.056 |
| Decode   |                      0.498 |          0.687 |
| Filter   |                    100.006 |          9.820 |
| Pipeline |                    100.553 |          4.975 |

#### Edge detection

Batch sweep (CPU with all threads, GPU). Memory is the peak resident memory of the process:

| Batch | CPU time (s) | CPU memory (GB) | GPU time (s) | GPU memory (GB) | GPU speedup |
| ----- | -----------: | --------------: | -----------: | --------------: | ----------: |
| 4     |        4.221 |            0.23 |        1.754 |            0.35 |       2.41× |
| 16    |        3.107 |            0.27 |        0.786 |            0.52 |       3.95× |
| 64    |        2.803 |            0.46 |        0.596 |            0.75 |       4.70× |
| 256   |        2.681 |            0.99 |        0.656 |            1.04 |       4.09× |

Thread sweep (CPU, batch of 64):

| Threads | Time (s) | Speedup | Efficiency | Throughput (Mpx/s) |
| ------- | -------: | ------: | ---------: | -----------------: |
| 1       |   21.053 |   1.00× |       100% |               33.7 |
| 2       |   10.573 |   1.99× |       100% |               67.2 |
| 4       |    5.538 |   3.80× |        95% |              128.3 |
| 8       |    3.124 |   6.74× |        84% |              227.4 |
| 16      |    2.709 |   7.77× |        49% |              262.2 |

- **Best CPU**: 2.681 s (16 threads, batch 256), 265.0 Mpx/s
- **Best GPU**: 0.596 s (batch 64), 1192.4 Mpx/s
- **GPU speedup**: 4.50× over the best CPU run, 35.33× over one thread

Stage times (median, s). The stages of different batches overlap, and on the GPU two images are in flight, so a stage can take longer than the pipeline:

| Stage    | CPU (16 threads, batch 256) | GPU (batch 64) |
| -------- | --------------------------: | -------------: |
| Read     |                       0.036 |          0.040 |
| Decode   |                       0.405 |          0.488 |
| Filter   |                       2.208 |          0.362 |
| Pipeline |                       2.681 |          0.596 |

**Observation**:

The GPU denoises the 500 radiographs (710 Mpx) in 5.0 s: 20× faster than the best CPU run (100 s) and 128× faster than one thread (637 s). The CPU gains nothing from the second hardware thread for denoising (6.36× on 16 threads, 6.31× on 8), while edge detection does (7.77× against 6.74×). Edge detection takes 0.60 s on the GPU, of which 0.49 s go to decoding the images.

### Chest X-ray (mixed)

500 images, 341 Mpx.

#### Denoising

Batch sweep (CPU with all threads, GPU). Memory is the peak resident memory of the process:

| Batch | CPU time (s) | CPU memory (GB) | GPU time (s) | GPU memory (GB) | GPU speedup |
| ----- | -----------: | --------------: | -----------: | --------------: | ----------: |
| 4     |       51.427 |            0.19 |        2.756 |            0.34 |      18.66× |
| 16    |       48.659 |            0.24 |        2.358 |            0.50 |      20.63× |
| 64    |       47.934 |            0.34 |        2.389 |            0.83 |      20.06× |
| 256   |       47.821 |            0.79 |        2.600 |            1.33 |      18.39× |

Thread sweep (CPU, batch of 64):

| Threads | Time (s) | Speedup | Efficiency | Throughput (Mpx/s) |
| ------- | -------: | ------: | ---------: | -----------------: |
| 1       |  306.270 |   1.00× |       100% |                1.1 |
| 2       |  153.579 |   1.99× |       100% |                2.2 |
| 4       |   83.419 |   3.67× |        92% |                4.1 |
| 8       |   47.961 |   6.39× |        80% |                7.1 |
| 16      |   47.621 |   6.43× |        40% |                7.2 |

- **Best CPU**: 47.621 s (16 threads, batch 64), 7.2 Mpx/s
- **Best GPU**: 2.358 s (batch 16), 144.8 Mpx/s
- **GPU speedup**: 20.19× over the best CPU run, 129.86× over one thread

Stage times (median, s). The stages of different batches overlap, and on the GPU two images are in flight, so a stage can take longer than the pipeline:

| Stage    | CPU (16 threads, batch 64) | GPU (batch 16) |
| -------- | -------------------------: | -------------: |
| Read     |                      0.017 |          0.036 |
| Decode   |                      0.256 |          0.423 |
| Filter   |                     47.653 |          4.772 |
| Pipeline |                     47.934 |          2.358 |

#### Edge detection

Batch sweep (CPU with all threads, GPU). Memory is the peak resident memory of the process:

| Batch | CPU time (s) | CPU memory (GB) | GPU time (s) | GPU memory (GB) | GPU speedup |
| ----- | -----------: | --------------: | -----------: | --------------: | ----------: |
| 4     |        2.154 |            0.18 |        1.023 |            0.32 |       2.11× |
| 16    |        1.558 |            0.23 |        0.500 |            0.43 |       3.11× |
| 64    |        1.354 |            0.33 |        0.332 |            0.59 |       4.07× |
| 256   |        1.308 |            0.58 |        0.364 |            0.66 |       3.59× |

Thread sweep (CPU, batch of 64):

| Threads | Time (s) | Speedup | Efficiency | Throughput (Mpx/s) |
| ------- | -------: | ------: | ---------: | -----------------: |
| 1       |   10.226 |   1.00× |       100% |               33.4 |
| 2       |    5.141 |   1.99× |        99% |               66.4 |
| 4       |    2.656 |   3.85× |        96% |              128.5 |
| 8       |    1.487 |   6.88× |        86% |              229.6 |
| 16      |    1.301 |   7.86× |        49% |              262.4 |

- **Best CPU**: 1.301 s (16 threads, batch 64), 262.4 Mpx/s
- **Best GPU**: 0.332 s (batch 64), 1027.1 Mpx/s
- **GPU speedup**: 3.91× over the best CPU run, 30.76× over one thread

Stage times (median, s). The stages of different batches overlap, and on the GPU two images are in flight, so a stage can take longer than the pipeline:

| Stage    | CPU (16 threads, batch 64) | GPU (batch 64) |
| -------- | -------------------------: | -------------: |
| Read     |                      0.017 |          0.023 |
| Decode   |                      0.255 |          0.252 |
| Filter   |                      1.058 |          0.212 |
| Pipeline |                      1.354 |          0.332 |

**Observation**:

The smaller images (0.68 Mpx on average) give the same throughput as the larger ones: 144.8 Mpx/s on the GPU and 7.2 Mpx/s on the CPU with 16 threads. Small batches suit the GPU denoising (2.36 s at batch 16, 2.76 s at batch 4).

### AAPM Low-Dose CT

500 images, 131 Mpx.

#### Denoising

Batch sweep (CPU with all threads, GPU). Memory is the peak resident memory of the process:

| Batch | CPU time (s) | CPU memory (GB) | GPU time (s) | GPU memory (GB) | GPU speedup |
| ----- | -----------: | --------------: | -----------: | --------------: | ----------: |
| 4     |       20.320 |            0.06 |        1.303 |            0.20 |      15.60× |
| 16    |       18.069 |            0.06 |        0.883 |            0.25 |      20.47× |
| 64    |       17.961 |            0.09 |        0.895 |            0.39 |      20.07× |
| 256   |       17.930 |            0.30 |        0.955 |            0.76 |      18.78× |

Thread sweep (CPU, batch of 64):

| Threads | Time (s) | Speedup | Efficiency | Throughput (Mpx/s) |
| ------- | -------: | ------: | ---------: | -----------------: |
| 1       |  116.076 |   1.00× |       100% |                1.1 |
| 2       |   58.115 |   2.00× |       100% |                2.3 |
| 4       |   31.594 |   3.67× |        92% |                4.1 |
| 8       |   17.951 |   6.47× |        81% |                7.3 |
| 16      |   17.925 |   6.48× |        40% |                7.3 |

- **Best CPU**: 17.925 s (16 threads, batch 64), 7.3 Mpx/s
- **Best GPU**: 0.883 s (batch 16), 148.5 Mpx/s
- **GPU speedup**: 20.31× over the best CPU run, 131.51× over one thread

Stage times (median, s). The stages of different batches overlap, and on the GPU two images are in flight, so a stage can take longer than the pipeline:

| Stage    | CPU (16 threads, batch 64) | GPU (batch 16) |
| -------- | -------------------------: | -------------: |
| Read     |                      0.007 |          0.019 |
| Decode   |                      0.087 |          0.095 |
| Filter   |                     17.841 |          1.705 |
| Pipeline |                     17.961 |          0.883 |

#### Edge detection

Batch sweep (CPU with all threads, GPU). Memory is the peak resident memory of the process:

| Batch | CPU time (s) | CPU memory (GB) | GPU time (s) | GPU memory (GB) | GPU speedup |
| ----- | -----------: | --------------: | -----------: | --------------: | ----------: |
| 4     |        0.801 |            0.06 |        0.409 |            0.19 |       1.96× |
| 16    |        0.539 |            0.06 |        0.208 |            0.22 |       2.59× |
| 64    |        0.514 |            0.06 |        0.172 |            0.26 |       2.99× |
| 256   |        0.504 |            0.14 |        0.185 |            0.33 |       2.72× |

Thread sweep (CPU, batch of 64):

| Threads | Time (s) | Speedup | Efficiency | Throughput (Mpx/s) |
| ------- | -------: | ------: | ---------: | -----------------: |
| 1       |    4.207 |   1.00× |       100% |               31.2 |
| 2       |    2.123 |   1.98× |        99% |               61.7 |
| 4       |    1.089 |   3.86× |        97% |              120.4 |
| 8       |    0.598 |   7.03× |        88% |              219.0 |
| 16      |    0.490 |   8.59× |        54% |              267.5 |

- **Best CPU**: 0.490 s (16 threads, batch 64), 267.5 Mpx/s
- **Best GPU**: 0.172 s (batch 64), 762.8 Mpx/s
- **GPU speedup**: 2.85× over the best CPU run, 24.48× over one thread

Stage times (median, s). The stages of different batches overlap, and on the GPU two images are in flight, so a stage can take longer than the pipeline:

| Stage    | CPU (16 threads, batch 64) | GPU (batch 64) |
| -------- | -------------------------: | -------------: |
| Read     |                      0.006 |          0.010 |
| Decode   |                      0.084 |          0.093 |
| Filter   |                      0.416 |          0.117 |
| Pipeline |                      0.514 |          0.172 |

**Observation**:

The 512×512 slices take 0.88 s on the GPU (148.5 Mpx/s) and 17.9 s on the CPU. Edge detection is only 2.9× faster on the GPU (0.17 s against 0.49 s): the work is so small that decoding (0.09 s of the 0.17 s) weighs as much as the filter (0.12 s).

### 2DeteCT

500 images, 524 Mpx.

#### Denoising

Batch sweep (CPU with all threads, GPU). Memory is the peak resident memory of the process:

| Batch | CPU time (s) | CPU memory (GB) | GPU time (s) | GPU memory (GB) | GPU speedup |
| ----- | -----------: | --------------: | -----------: | --------------: | ----------: |
| 4     |       79.308 |            0.08 |        3.516 |            0.30 |      22.55× |
| 16    |       75.553 |            0.17 |        3.519 |            0.50 |      21.47× |
| 64    |       75.226 |            0.41 |        3.594 |            1.21 |      20.93× |
| 256   |       75.079 |            1.46 |        3.955 |            2.98 |      18.98× |

Thread sweep (CPU, batch of 64):

| Threads | Time (s) | Speedup | Efficiency | Throughput (Mpx/s) |
| ------- | -------: | ------: | ---------: | -----------------: |
| 1       |  490.159 |   1.00× |       100% |                1.1 |
| 2       |  245.261 |   2.00× |       100% |                2.1 |
| 4       |  132.693 |   3.69× |        92% |                4.0 |
| 8       |   76.969 |   6.37× |        80% |                6.8 |
| 16      |   74.982 |   6.54× |        41% |                7.0 |

- **Best CPU**: 74.982 s (16 threads, batch 64), 7.0 Mpx/s
- **Best GPU**: 3.516 s (batch 4), 149.1 Mpx/s
- **GPU speedup**: 21.32× over the best CPU run, 139.40× over one thread

Stage times (median, s). The stages of different batches overlap, and on the GPU two images are in flight, so a stage can take longer than the pipeline:

| Stage    | CPU (16 threads, batch 64) | GPU (batch 4) |
| -------- | -------------------------: | ------------: |
| Read     |                      0.019 |         0.063 |
| Decode   |                      0.307 |         1.126 |
| Filter   |                     74.864 |         7.210 |
| Pipeline |                     75.226 |         3.516 |

#### Edge detection

Batch sweep (CPU with all threads, GPU). Memory is the peak resident memory of the process:

| Batch | CPU time (s) | CPU memory (GB) | GPU time (s) | GPU memory (GB) | GPU speedup |
| ----- | -----------: | --------------: | -----------: | --------------: | ----------: |
| 4     |        3.560 |            0.06 |        1.207 |            0.23 |       2.95× |
| 16    |        2.526 |            0.08 |        0.441 |            0.30 |       5.73× |
| 64    |        2.371 |            0.17 |        0.400 |            0.48 |       5.93× |
| 256   |        2.286 |            0.54 |        0.455 |            0.75 |       5.03× |

Thread sweep (CPU, batch of 64):

| Threads | Time (s) | Speedup | Efficiency | Throughput (Mpx/s) |
| ------- | -------: | ------: | ---------: | -----------------: |
| 1       |   18.580 |   1.00× |       100% |               28.2 |
| 2       |    9.352 |   1.99× |        99% |               56.1 |
| 4       |    4.824 |   3.85× |        96% |              108.7 |
| 8       |    2.674 |   6.95× |        87% |              196.1 |
| 16      |    2.266 |   8.20× |        51% |              231.4 |

- **Best CPU**: 2.266 s (16 threads, batch 64), 231.4 Mpx/s
- **Best GPU**: 0.400 s (batch 64), 1311.8 Mpx/s
- **GPU speedup**: 5.67× over the best CPU run, 46.49× over one thread

Stage times (median, s). The stages of different batches overlap, and on the GPU two images are in flight, so a stage can take longer than the pipeline:

| Stage    | CPU (16 threads, batch 64) | GPU (batch 64) |
| -------- | -------------------------: | -------------: |
| Read     |                      0.021 |          0.022 |
| Decode   |                      0.315 |          0.343 |
| Filter   |                      1.992 |          0.236 |
| Pipeline |                      2.371 |          0.400 |

**Observation**:

The CPU denoising scales 6.5× on 16 threads (6.4× on 8): the machine has 8 physical cores, so the second hardware thread of each adds almost nothing. The GPU is 21× faster than the best CPU run and 139× faster than one thread, at 149 Mpx/s. It is fastest with small batches (3.52 s at batch 4 and 16); batch 256 is 12% slower and needs ten times the memory of batch 4. Edge detection scales better on the CPU (8.2× on 16 threads) and is 5.7× faster on the GPU, where 0.34 s of the 0.40 s go to decoding the PNG files.

### Overall observations

- Denoising costs the same per pixel on every dataset, whatever the noise level, the strength or the image size: 1.1 Mpx/s on one thread, 7.0–7.3 Mpx/s on 16 threads, and 143–149 Mpx/s on the GPU. The GPU is 20–21× faster than the best CPU run and 128–139× faster than one thread.
- Edge detection on the GPU is 2.9–5.7× faster than on 16 threads. Decoding the images takes about as long as the filter itself, so it limits the speedup, most on the smallest images.
- The CPU has 8 physical cores: the second hardware thread adds 0.1–2.7% for denoising and 14–22% for edge detection.
- The GPU is fastest with batches of 4–16 images for denoising and of 64 for edge detection. A batch of 256 makes denoising 8–13% slower and needs 3–6 times the memory of a batch of 16; the CPU denoising time hardly depends on the batch size above 16.
- Denoising raises the PSNR by 4.5 to 9.2 dB on average, and only 4 of the 36,669 images get worse (by at most 1.6 dB). The noise added to the chest X-rays is estimated to within 5%.
- The edge F1 rises on every dataset: from 0.69–0.74 to 0.80 on the chest X-rays, from 0.82 to 0.85 on AAPM, and from 0.25 to 0.75 on 2DeteCT. Where the noisy image is already reliable, denoising can lower it: on the cleanest quarter of the AAPM slices (−0.014) and of the mixed chest X-rays (35% of the images).
- The 500-image samples represent the full datasets to within 0.13 dB and 0.003 edge F1.
- The CPU and GPU outputs are byte-identical in all eight runs, for denoising and for denoising with edges.

## Runs

| Run                       | Dataset                 | Kind                                | Images | Duration (min) |
| ------------------------- | ----------------------- | ----------------------------------- | -----: | -------------: |
| chest-xrau-pneumonia-full | Chest X-ray (pneumonia) | quality, all images                 |  5,856 |           46.0 |
| chest-xrau-pneumonia-500  | Chest X-ray (pneumonia) | speed and quality, 500-image sample |    500 |          125.3 |
| chest-xray-full           | Chest X-ray (mixed)     | quality, all images                 | 25,553 |          104.4 |
| chest-xray-500            | Chest X-ray (mixed)     | speed and quality, 500-image sample |    500 |           60.2 |
| aapm-full                 | AAPM Low-Dose CT        | quality, all images                 |  4,260 |            6.0 |
| aapm-500                  | AAPM Low-Dose CT        | speed and quality, 500-image sample |    500 |           22.8 |
| 2detect-full              | 2DeteCT                 | quality, all images                 |  1,000 |            5.9 |
| 2detect-500               | 2DeteCT                 | speed and quality, 500-image sample |    500 |           95.7 |
