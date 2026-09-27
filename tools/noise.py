#!/usr/bin/env python3
"""
noise.py - Add artificial noise to images, to test denoising.

Takes an image or a directory of images and writes noisy copies, together
with a manifest (noise.csv) recording, for every image, the noise parameters
and the standard deviation of the noise actually added, which is the ground
truth for evaluating a denoiser (see compare.py) and imgfilter's noise
estimate.

Noise models:

  poisson   Signal-dependent photon-counting noise, as in low-dose
            acquisitions: a pixel of normalized brightness v receives on
            average v * photons * dose photons, the count is drawn from a
            Poisson distribution, and scaled back. Lower dose means fewer
            photons and stronger noise, strongest in dark regions. The dose
            is fixed (--dose) or drawn per image from [--min-dose,
            --max-dose]. Note: this treats brightness as proportional to the
            photon count; in a radiograph displayed with bone white it is the
            other way round, so this is a model of signal-dependent noise,
            not a physical low-dose simulation.

  gaussian  Additive white Gaussian noise of standard deviation --sigma
            (in gray levels of an 8-bit image), the standard setting in the
            denoising literature.

Output is always lossless (PNG, or PGM with --format pgm): saving noisy
images as JPEG would partly remove the noise again and add compression
artifacts. Output files keep the input's base name. 16-bit inputs stay
16-bit.

Every image's noise is drawn from a generator seeded with --seed and the
file name, so a file always gets the same noise, whatever the order or the
subset of files processed.

Input and output follow imgfilter's conventions:
  file -> file, file -> directory, directory -> directory (created).

Requires numpy and Pillow (Arch: python-numpy python-pillow).

Examples:
  tools/noise.py data/clean data/noisy                      # dose 10-50%
  tools/noise.py --dose 0.2 data/clean data/noisy_20
  tools/noise.py --model gaussian --sigma 20 xray.png noisy.png
"""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ImportError as e:
    sys.exit(f"noise.py: {e.name} is required (Arch: python-numpy python-pillow)")

EXTENSIONS = {".png", ".jpg", ".jpeg", ".pgm", ".pnm", ".bmp", ".tga", ".tif", ".tiff"}


def load_gray(path):
    """Loads an image as grayscale, keeping 16-bit data 16-bit."""
    img = Image.open(path)
    if img.mode in ("I;16", "I;16B", "I;16L"):
        a = np.array(img, dtype=np.uint16)
    elif img.mode == "I":
        a = np.clip(np.array(img), 0, 65535).astype(np.uint16)
    else:
        a = np.array(img.convert("L"), dtype=np.uint8)
    return a


def rng_for(seed, name):
    """Random generator for one file: seeded with the seed and the file name."""
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def add_poisson(a, rng, dose, photons):
    """Photon-counting noise; returns the noisy image, same dtype."""
    peak = np.iinfo(a.dtype).max
    expected = a.astype(np.float64) / peak * photons * dose
    noisy = rng.poisson(expected) / (photons * dose) * peak
    return np.clip(np.rint(noisy), 0, peak).astype(a.dtype)


def add_gaussian(a, rng, sigma):
    """Additive white Gaussian noise; sigma in 8-bit gray levels."""
    peak = np.iinfo(a.dtype).max
    scale = peak / 255.0
    noisy = a.astype(np.float64) + rng.normal(0.0, sigma * scale, a.shape)
    return np.clip(np.rint(noisy), 0, peak).astype(a.dtype)


def save(a, path):
    """Saves losslessly; PGM or PNG by the file extension."""
    Image.fromarray(a).save(path)


def list_inputs(src):
    if src.is_dir():
        files = sorted(p for p in src.iterdir()
                       if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in EXTENSIONS)
        if not files:
            sys.exit(f"noise.py: no images found in {src}")
        return files
    if src.is_file():
        return [src]
    sys.exit(f"noise.py: cannot access {src}")


def main():
    ap = argparse.ArgumentParser(
        description="Add artificial noise to an image or a directory of images.",
        epilog="Output is lossless (PNG, or PGM); a manifest noise.csv records the "
               "parameters and the added noise level of every image.")
    ap.add_argument("input", type=Path, help="image or directory of images")
    ap.add_argument("output", type=Path, help="output image or directory")
    ap.add_argument("--model", choices=("poisson", "gaussian"), default="poisson",
                    help="noise model (default: poisson)")
    ap.add_argument("--dose", type=float, help="poisson: fixed dose fraction, e.g. 0.2")
    ap.add_argument("--min-dose", type=float, default=0.10,
                    help="poisson: smallest random dose fraction (default: 0.10)")
    ap.add_argument("--max-dose", type=float, default=0.50,
                    help="poisson: largest random dose fraction (default: 0.50)")
    ap.add_argument("--photons", type=float, default=1000.0,
                    help="poisson: photons of the brightest pixel at full dose (default: 1000)")
    ap.add_argument("--sigma", type=float, default=20.0,
                    help="gaussian: standard deviation in 8-bit gray levels (default: 20)")
    ap.add_argument("--format", choices=("png", "pgm"), default="png",
                    help="output format (default: png)")
    ap.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    args = ap.parse_args()

    if args.model == "poisson":
        lo, hi = (args.dose, args.dose) if args.dose is not None else (args.min_dose, args.max_dose)
        if not 0 < lo <= hi:
            sys.exit("noise.py: doses must satisfy 0 < min-dose <= max-dose")
        if args.photons <= 0:
            sys.exit("noise.py: --photons must be positive")
    elif args.sigma < 0:
        sys.exit("noise.py: --sigma must not be negative")

    files = list_inputs(args.input)

    # output: a directory for a directory input, or when it is/looks like one
    to_dir = args.input.is_dir() or args.output.is_dir() or str(args.output).endswith("/")
    if args.input.is_dir() and args.output.exists() and not args.output.is_dir():
        sys.exit(f"noise.py: cannot write a directory of images into file {args.output}")
    if to_dir:
        args.output.mkdir(parents=True, exist_ok=True)
        if args.output.resolve() == (args.input.resolve() if args.input.is_dir() else args.input.parent.resolve()):
            sys.exit("noise.py: output directory must differ from the input directory")
        manifest = args.output / "noise.csv"
    else:
        manifest = args.output.with_name(args.output.stem + ".noise.csv")

    ext = "." + args.format
    rows = []
    for i, src in enumerate(files, 1):
        a = load_gray(src)
        rng = rng_for(args.seed, src.name)

        if args.model == "poisson":
            dose = rng.uniform(lo, hi) if lo < hi else lo
            noisy = add_poisson(a, rng, dose, args.photons)
            param = f"dose={dose:.4f}"
        else:
            dose = None
            noisy = add_gaussian(a, rng, args.sigma)
            param = f"sigma={args.sigma:g}"

        dst = args.output / (src.stem + ext) if to_dir else args.output
        save(noisy, dst)

        # noise actually added, in 8-bit gray levels
        scale = 255.0 / np.iinfo(a.dtype).max
        added = (noisy.astype(np.float64) - a.astype(np.float64)) * scale
        rows.append({
            "file": dst.name,
            "source": str(src),
            "model": args.model,
            "dose": f"{dose:.6f}" if dose is not None else "",
            "photons": f"{args.photons:g}" if args.model == "poisson" else "",
            "sigma": f"{args.sigma:g}" if args.model == "gaussian" else "",
            "noise_std": f"{added.std():.4f}",
        })
        print(f"[{i:{len(str(len(files)))}d}/{len(files)}] {src.name} -> {dst.name} "
              f"({param}, added noise std {added.std():.2f})")

    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
