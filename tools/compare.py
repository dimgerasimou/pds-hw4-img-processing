#!/usr/bin/env python3
"""
compare.py - PSNR of processed images against clean references.

Compares one or more images or directories against a reference (the clean
images): files are matched by base name, whatever their extension (so
clean/scan01.jpeg matches denoised/scan01.pgm). For every compared directory
it prints the mean, minimum and maximum PSNR; with --per-image, every image.

Typical use, with noise.py:

  tools/noise.py data/clean data/noisy
  ./bin/imgfilter -d -o data/denoised data/noisy
  tools/compare.py data/clean data/noisy data/denoised

PSNR = 10 log10(peak^2 / MSE), on the 8-bit scale (16-bit images are
scaled to 8 bits). Higher is better; +1 dB is a clearly visible difference
at these noise levels.

Requires numpy and Pillow (Arch: python-numpy python-pillow).
"""

import argparse
import math
import sys
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ImportError as e:
    sys.exit(f"compare.py: {e.name} is required (Arch: python-numpy python-pillow)")

EXTENSIONS = {".png", ".jpg", ".jpeg", ".pgm", ".pnm", ".bmp", ".tga", ".tif", ".tiff"}


def load8(path):
    """Loads an image as 8-bit-scale grayscale floats."""
    img = Image.open(path)
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        return np.array(img, dtype=np.float64) * (255.0 / 65535.0)
    return np.array(img.convert("L"), dtype=np.float64)


def images(path):
    """Base name -> file, for a directory or a single file."""
    if path.is_dir():
        return {p.stem: p for p in sorted(path.iterdir())
                if p.is_file() and p.suffix.lower() in EXTENSIONS}
    if path.is_file():
        return {path.stem: path}
    sys.exit(f"compare.py: cannot access {path}")


def psnr(ref, img):
    mse = np.mean((ref - img) ** 2)
    return math.inf if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser(description="PSNR of images against clean references.")
    ap.add_argument("reference", type=Path, help="clean image or directory")
    ap.add_argument("test", type=Path, nargs="+", help="images or directories to compare")
    ap.add_argument("--per-image", action="store_true", help="print the PSNR of every image")
    args = ap.parse_args()

    ref = images(args.reference)
    print(f"{'set':30s} {'images':>7s} {'mean dB':>9s} {'min dB':>8s} {'max dB':>8s}")

    for t in args.test:
        values = []
        for stem, path in images(t).items():
            if stem not in ref:
                continue
            a, b = load8(ref[stem]), load8(path)
            if a.shape != b.shape:
                print(f"  {path.name}: size {b.shape} differs from reference {a.shape}, skipped")
                continue
            p = psnr(a, b)
            values.append(p)
            if args.per_image:
                print(f"  {path.name:40s} {p:8.2f} dB")

        if not values:
            print(f"{str(t):30s} {'0':>7s}   (no image matches the reference names)")
            continue

        finite = [v for v in values if math.isfinite(v)]
        mean = sum(finite) / len(finite) if finite else math.inf
        print(f"{str(t):30s} {len(values):7d} {mean:9.2f} {min(values):8.2f} {max(values):8.2f}")


if __name__ == "__main__":
    main()
