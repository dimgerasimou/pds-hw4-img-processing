#!/usr/bin/env python3
"""
convert.py - Convert .npy and .tif images to 8-bit PNG with one fixed window.

imgfilter reads 8-bit images, but CT data comes as float or integer arrays. Each
value v becomes round(255 * (v - LO) / (HI - LO)), clipped to 0..255, with the
same LO and HI for every image of both sets: scaling each image by its own
minimum and maximum would give a noisy image and its clean counterpart different
gray levels, and PSNR would measure that instead of the denoising.

Layouts:
  ROOT/fd and ROOT/qd (AAPM mirror: full and quarter dose, same file names):
      written to OUTPUT/reference and OUTPUT/input
  ROOT/<name>/ per image, with --reference-file and --input-file naming the two
  files inside (2DeteCT: mode2/reconstruction.tif, mode1/reconstruction.tif):
      written to OUTPUT/reference and OUTPUT/input as <name>.png
  ROOT with image files: written to OUTPUT

The window is --window LO HI in the units of the data, or the percentiles
(--percentiles, default 0.5 99.5) of the reference set, applied to both.
--info prints the value ranges and stops. Clipping removes noise at the window's
edges, so a soft-tissue window such as -160 240 (HU) usually beats the percentiles.

Examples:
  tools/convert.py --info aapm/data/test
  tools/convert.py --window -160 240 aapm/data/test aapm_png
  tools/convert.py --reference-file mode2/reconstruction.tif \\
      --input-file mode1/reconstruction.tif --limit 200 2detect 2detect_png
"""

import argparse
import re
import sys
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ImportError as e:
    sys.exit(f"convert.py: {e.name} is required (Arch: python-numpy python-pillow)")

SAMPLE = 4096   # pixels per image that enter the percentiles
EXTENSIONS = {".npy", ".tif", ".tiff"}


def natural(p):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def image_files(d):
    return sorted((p for p in Path(d).iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS),
                  key=natural)


def read(path):
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    return np.array(Image.open(path))


def load(path):
    a = read(path)
    img = np.squeeze(a)
    if img.ndim != 2:
        sys.exit(f"convert.py: {path}: expected one 2D image, the array has shape {a.shape}")
    if not np.isfinite(img).all():
        sys.exit(f"convert.py: {path}: contains NaN or infinity")
    return img.astype(np.float64)


def scan(items):
    """Value range and a pooled random sample of the pixels of a set of images."""
    rng = np.random.default_rng(0)
    lo, hi, sample = np.inf, -np.inf, []
    first = read(items[0][1])
    for _, f in items:
        a = load(f)
        lo, hi = min(lo, a.min()), max(hi, a.max())
        sample.append(a.ravel()[rng.integers(0, a.size, SAMPLE)])
    return {"dtype": first.dtype, "shape": np.squeeze(first).shape, "n": len(items),
            "min": lo, "max": hi, "sample": np.concatenate(sample)}


def describe(name, s):
    q = np.percentile(s["sample"], [0.1, 1, 50, 99, 99.9])
    print(f"{name:10s} {s['n']} images, {s['dtype']}, {s['shape'][1]}x{s['shape'][0]}, "
          f"min {s['min']:.6g}, max {s['max']:.6g}")
    print(f"{'':10s} percentiles 0.1 / 1 / 50 / 99 / 99.9: " + " / ".join(f"{v:.6g}" for v in q))


def to_u8(a, lo, hi):
    return np.clip(np.rint((a - lo) * (255.0 / (hi - lo))), 0, 255).astype(np.uint8)


def convert(items, out, lo, hi):
    """Writes the PNGs; returns the fractions of pixels clipped below and above the window."""
    out.mkdir(parents=True)
    below = above = total = 0
    for name, f in items:
        a = load(f)
        below += int((a < lo).sum())
        above += int((a > hi).sum())
        total += a.size
        Image.fromarray(to_u8(a, lo, hi)).save(out / (name + ".png"), compress_level=1)
    return below / total, above / total


def main():
    ap = argparse.ArgumentParser(
        description="Convert .npy and .tif images to 8-bit PNG with one fixed window.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("Layouts:")[1])
    ap.add_argument("input", type=Path, help="directory with fd/ and qd/, with one subdirectory "
                                             "per image, or with image files")
    ap.add_argument("output", type=Path, nargs="?", help="output directory (must not exist)")
    ap.add_argument("--reference-sub", default="fd", help="subdirectory with the reference (default: fd)")
    ap.add_argument("--input-sub", default="qd", help="subdirectory with the noisy input (default: qd)")
    ap.add_argument("--reference-file", help="path of the reference inside each subdirectory of ROOT")
    ap.add_argument("--input-file", help="path of the noisy input inside each subdirectory of ROOT")
    ap.add_argument("--window", type=float, nargs=2, metavar=("LO", "HI"),
                    help="window in the units of the data")
    ap.add_argument("--percentiles", type=float, nargs=2, default=(0.5, 99.5), metavar=("LO", "HI"),
                    help="window from these percentiles of the reference set (default: 0.5 99.5)")
    ap.add_argument("--limit", type=int, help="convert only the first N images (of the pairs)")
    ap.add_argument("--info", action="store_true", help="print the value ranges and stop")
    args = ap.parse_args()

    if not args.input.is_dir():
        ap.error(f"{args.input} is not a directory")
    if not args.info and not args.output:
        ap.error("an output directory is needed (or --info)")
    if args.output and args.output.exists():
        sys.exit(f"convert.py: {args.output} exists; remove it or choose another name")
    if args.window and args.window[0] >= args.window[1]:
        ap.error("--window: LO must be smaller than HI")
    if not 0 <= args.percentiles[0] < args.percentiles[1] <= 100:
        ap.error("--percentiles: need 0 <= LO < HI <= 100")

    if bool(args.reference_file) != bool(args.input_file):
        ap.error("--reference-file and --input-file go together")

    ref_dir, in_dir = args.input / args.reference_sub, args.input / args.input_sub
    if args.reference_file:
        subs = sorted((p for p in args.input.iterdir() if p.is_dir()), key=natural)
        pairs = [(p.name, p / args.reference_file, p / args.input_file) for p in subs]
        pairs = [t for t in pairs if t[1].is_file() and t[2].is_file()]
        if not pairs:
            sys.exit(f"convert.py: no subdirectory of {args.input} has both {args.reference_file} and "
                     f"{args.input_file}" + (f" (it has {len(subs)} subdirectories, the first is "
                                             f"{subs[0].name}/)" if subs else ""))
        pairs = pairs[:args.limit] if args.limit else pairs
        sets = [("reference", [(n, r) for n, r, _ in pairs], args.output and args.output / "reference"),
                ("input", [(n, i) for n, _, i in pairs], args.output and args.output / "input")]
    elif ref_dir.is_dir() and in_dir.is_dir():
        ref = {f.stem: f for f in image_files(ref_dir)}
        inp = {f.stem: f for f in image_files(in_dir)}
        names = sorted(ref.keys() & inp.keys(), key=lambda n: natural(Path(n)))
        if not names:
            sys.exit(f"convert.py: no file name is in both {ref_dir} and {in_dir}")
        if len(names) < max(len(ref), len(inp)):
            print(f"note: {len(ref) - len(names)} reference and {len(inp) - len(names)} input files "
                  "have no partner, ignored")
        names = names[:args.limit] if args.limit else names
        sets = [("reference", [(n, ref[n]) for n in names], args.output and args.output / "reference"),
                ("input", [(n, inp[n]) for n in names], args.output and args.output / "input")]
    else:
        files = image_files(args.input)
        if not files:
            subs = sorted(p.name for p in args.input.iterdir() if p.is_dir())
            sys.exit(f"convert.py: no image files and no {args.reference_sub}/ + {args.input_sub}/ in "
                     f"{args.input}" + (f"; it has subdirectories {', '.join(subs[:8])}"
                                        f"{', ...' if len(subs) > 8 else ''}; try one of them, or give "
                                        "--reference-file and --input-file" if subs else ""))
        files = files[:args.limit] if args.limit else files
        sets = [("images", [(f.stem, f) for f in files], args.output)]

    stats = {name: scan(items) for name, items, _ in sets}
    for name, s in stats.items():
        describe(name, s)

    ref_sample = stats[sets[0][0]]["sample"]
    if args.window:
        lo, hi = args.window
        how = "given"
    else:
        lo, hi = (float(f"{v:.6g}") for v in np.percentile(ref_sample, args.percentiles))  # as printed
        how = f"{args.percentiles[0]:g}-{args.percentiles[1]:g} percentiles of the {sets[0][0]} set"
    if not hi > lo:
        sys.exit("convert.py: the window is empty; give one with --window")
    print(f"window     {lo:.6g} .. {hi:.6g} ({how}), the same for every image")

    if len(sets) == 2:
        shift = abs(np.median(stats["input"]["sample"]) - np.median(ref_sample))
        if shift > 0.05 * (hi - lo):
            print("warning: the median values of the two sets differ by "
                  f"{shift:.6g}, more than 5% of the window; do both use the same units?")

    if args.info:
        return
    for name, items, out in sets:
        below, above = convert(items, out, lo, hi)
        print(f"{name:10s} wrote {len(items)} PNGs to {out} "
              f"(clipped: {100 * below:.2f}% below, {100 * above:.2f}% above the window)")
    if len(sets) == 2:
        print(f"next: tools/bench.py --reference {args.output / 'reference'} "
              f"--skip batch,threads {args.output / 'input'}")


if __name__ == "__main__":
    main()
