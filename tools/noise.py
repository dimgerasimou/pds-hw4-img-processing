#!/usr/bin/env python3
"""
noise.py - Write noisy copies of a directory of images, to test denoising.

  poisson   signal-dependent photon-counting noise; the dose is --dose, or drawn
            per image from 10-50%. A model of signal-dependent noise, not a
            physical low-dose simulation.
  gaussian  additive white noise of standard deviation --sigma (8-bit gray levels)

Output is lossless PNG with the input's base names. noise.csv records the noise
actually added to every image, which bench.py compares with imgfilter's estimate.
Each image's noise depends only on --seed and its file name, so the result is the
same whatever --jobs is. --clean DIR also writes the originals there as PNG.

Examples:
  tools/noise.py data/clean data/noisy
  tools/noise.py --model gaussian --sigma 20 data/clean data/noisy20
"""

import argparse
import csv
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ImportError as e:
    sys.exit(f"noise.py: {e.name} is required (Arch: python-numpy python-pillow)")

EXTENSIONS = {".png", ".jpg", ".jpeg", ".pgm", ".pnm", ".bmp", ".tga", ".tif", ".tiff"}
DOSE_RANGE = (0.10, 0.50)
PHOTONS = 1000.0   # of the brightest pixel at full dose


def load_gray(path):
    """Grayscale, 16-bit data kept 16-bit."""
    img = Image.open(path)
    if img.mode in ("I;16", "I;16B", "I;16L"):
        return np.array(img, dtype=np.uint16)
    if img.mode == "I":
        return np.clip(np.array(img), 0, 65535).astype(np.uint16)
    return np.array(img.convert("L"), dtype=np.uint8)


def rng_for(seed, name):
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def process(job):
    """Noisy copy of one image; returns its manifest row and a description of the noise."""
    src, dst, clean, model, dose, sigma, seed = job
    a = load_gray(src)
    if clean:
        Image.fromarray(a).save(clean / dst.name, compress_level=1)
    peak = float(np.iinfo(a.dtype).max)
    rng = rng_for(seed, dst.name)
    if model == "poisson":
        dose = dose if dose is not None else rng.uniform(*DOSE_RANGE)
        noisy = rng.poisson(a / peak * PHOTONS * dose) / (PHOTONS * dose) * peak
        param = f"dose={dose:.4f}"
    else:
        dose = None
        noisy = a.astype(np.float64) + rng.normal(0.0, sigma * peak / 255.0, a.shape)
        param = f"sigma={sigma:g}"
    noisy = np.clip(np.rint(noisy), 0, peak).astype(a.dtype)
    Image.fromarray(noisy).save(dst)

    added = (noisy.astype(np.float64) - a) * (255.0 / peak)
    row = {"file": dst.name, "model": model, "dose": "" if dose is None else f"{dose:.6f}",
           "sigma": f"{sigma:g}" if dose is None else "", "noise_std": f"{added.std():.4f}"}
    return row, f"{src.name}: {param}, added noise std {added.std():.2f}"


def main():
    ap = argparse.ArgumentParser(description="Write noisy copies of a directory of images.")
    ap.add_argument("input", type=Path, help="directory of images")
    ap.add_argument("output", type=Path, help="output directory")
    ap.add_argument("--model", choices=("poisson", "gaussian"), default="poisson")
    ap.add_argument("--dose", type=float, help="poisson: fixed dose fraction, e.g. 0.2")
    ap.add_argument("--sigma", type=float, default=20.0, help="gaussian: standard deviation (default: 20)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", type=Path, help="also write the originals here, as PNG")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1, help="processes (default: all cores)")
    args = ap.parse_args()

    if not args.input.is_dir():
        sys.exit(f"noise.py: {args.input} is not a directory")
    if args.dose is not None and args.dose <= 0:
        sys.exit("noise.py: --dose must be positive")
    if args.sigma < 0:
        sys.exit("noise.py: --sigma must not be negative")
    files = sorted(p for p in args.input.iterdir()
                   if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in EXTENSIONS)
    if not files:
        sys.exit(f"noise.py: no images in {args.input}")
    if args.output.resolve() == args.input.resolve():
        sys.exit("noise.py: the output directory must differ from the input directory")
    if args.jobs < 1:
        sys.exit("noise.py: --jobs must be at least 1")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.clean:
        args.clean.mkdir(parents=True, exist_ok=True)

    # seeded by the output name (x.png), which is what the noise of a JPEG original was always drawn from
    jobs = [(src, args.output / (src.stem + ".png"), args.clean, args.model, args.dose, args.sigma, args.seed)
            for src in files]
    rows = []
    if args.jobs == 1:
        results = map(process, jobs)
    else:
        results = ProcessPoolExecutor(args.jobs).map(process, jobs, chunksize=4)
    for i, (row, text) in enumerate(results, 1):
        rows.append(row)
        print(f"[{i}/{len(files)}] {text}")

    with open(args.output / "noise.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
