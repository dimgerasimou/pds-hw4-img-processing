#!/usr/bin/env python3
"""
tune.py - Find good NLM denoising settings for images with a known truth.

imgfilter estimates the noise level and picks the strength for every image by
itself, and this is right for most noise, including the spatially correlated
noise of CT reconstructions. tune.py runs a grid of settings on a small
sample of image pairs and prints the PSNR each one reaches, next to the
automatic setting, to show whether anything beats it for your data. The best
can be given to imgfilter, or to bench.py with --nlm "...".

Two scores are reported for every setting: the PSNR, and the edge F1, that is
how well the Canny edges of the denoised image agree with those of the truth
(1-pixel tolerance, as in bench.py). They can disagree: strong smoothing keeps
raising the PSNR but blurs fine edges, so the F1 peaks at a milder setting.
Sort by the one that matches what the images are for (--metric).

The settings (imgfilter options):
  -N <sigma>  noise level in gray levels, or "1.5x" for 1.5 times the
              automatic estimate (which follows the noise of every image)
  -H <k>      strength: the filter tolerates patch differences of about
              k * sigma; "auto" leaves it to imgfilter (0.7 to 1.6)
  -P <r>      patch radius
  -S <r>      search radius

Pairs are given as for bench.py: --reference DIR holds the truth, files match
by base name, or by --input-suffix / --reference-suffix, or through --pairs.
A few dozen pairs are enough. The best PSNR can favor smoother images than
the eye does, so look at the result, and check the chosen settings on other
images (bench.py --nlm "..." with another --seed).

Examples:
  tools/tune.py --reference data/full data/low
  tools/tune.py --reference data/full --sample 40 --sigma 0.7x,1x,1.5x --strength auto,1.2 data/low
  tools/tune.py --reference data/full --gpu --patch 2,3,4,5 data/low
  tools/tune.py --reference data/full --metric f1 --tiers 3 data/low
"""

import argparse
import itertools
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
sys.path.insert(0, str(TOOLS))
sys.dont_write_bytecode = True

try:
    import numpy as np
    import bench
    import compare
except ImportError as e:
    sys.exit(f"tune.py: {e.name} is required (Arch: python-numpy python-pillow)")


def str_list(s):
    return [x for x in s.replace(" ", ",").split(",") if x]


def int_list(s):
    return [int(x) for x in str_list(s)]


def psnrs(refs, outdir):
    """PSNR of each image in outdir against its reference (a dict of arrays)."""
    files = compare.images(outdir)
    return {k: compare.psnr(refs[k].astype(float), bench.gray(files[k]).astype(float)) for k in refs}


def run_edges(args, indir, outdir):
    cmd = [str(args.bin), "-e", "-B", "16"] + (["-g"] if args.gpu else []) + ["-o", str(outdir), str(indir)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"tune.py: imgfilter failed ({p.returncode}): {p.stderr.strip()[-300:]}")


def f1s(args, truth_edges, indir, work):
    """Edge F1 of each image in indir: its Canny edges against the truth's edges."""
    out = work / "edges"
    shutil.rmtree(out, ignore_errors=True)
    run_edges(args, indir, out)
    files = compare.images(out)
    return {k: bench.edge_f1(truth_edges[k], bench.gray(files[k])) for k in truth_edges}


def mean(d):
    return float(np.mean(list(d.values())))


def denoise(args, opts, indir, outdir, bench_json=None):
    cmd = [str(args.bin), "-d", *opts, "-B", "16"]
    if args.gpu:
        cmd.append("-g")
    if bench_json:
        cmd += ["-b", str(bench_json), "-n", "1"]
    p = subprocess.run(cmd + ["-o", str(outdir), str(indir)], capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"tune.py: imgfilter failed ({p.returncode}): {p.stderr.strip()[-300:]}")


def main():
    ap = argparse.ArgumentParser(description="Find good NLM denoising settings for images with a known truth.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("input", type=Path, help="directory of noisy images")
    ap.add_argument("--reference", type=Path, required=True, help="directory with the truth")
    ap.add_argument("--input-suffix", default="")
    ap.add_argument("--reference-suffix", default="")
    ap.add_argument("--pairs", type=Path, help='CSV of "input,reference" file names')
    ap.add_argument("--sample", type=int, default=20, help="image pairs to use (default: 20)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma", type=str_list, default=["0.7x", "1x", "1.4x"],
                    help='values of -N: numbers, or factors like 1.4x (default: 0.7x,1x,1.4x)')
    ap.add_argument("--strength", type=str_list, default=["auto", "0.7", "1.0", "1.4", "1.8"],
                    help="values of -H, or auto (default: auto,0.7,1.0,1.4,1.8)")
    ap.add_argument("--patch", type=int_list, default=[2, 3], help="values of -P (default: 2,3)")
    ap.add_argument("--search", type=int_list, default=[10], help="values of -S (default: 10)")
    ap.add_argument("--gpu", action="store_true", help="run imgfilter on the GPU (-g)")
    ap.add_argument("--metric", choices=("psnr", "f1"), default="psnr",
                    help="what to sort the settings by (default: psnr)")
    ap.add_argument("--tiers", type=int, default=1,
                    help="also report the best settings within this many noise levels, "
                         "noisiest first (default: 1, no split)")
    ap.add_argument("--top", type=int, default=8, help="settings to list (default: 8)")
    ap.add_argument("--bin", type=Path, default=REPO / "bin" / "imgfilter")
    args = ap.parse_args()

    if not args.bin.is_file():
        sys.exit(f"tune.py: {args.bin} not found; run make first")
    if not args.input.is_dir() or not args.reference.is_dir():
        ap.error("the input and --reference must be directories")
    if args.sample < 1:
        ap.error("--sample must be at least 1")
    if args.pairs and (args.input_suffix or args.reference_suffix):
        ap.error("--pairs and the suffix options are alternatives")

    pairs = bench.find_pairs(args)
    if len(pairs) > args.sample:
        pairs = sorted(random.Random(args.seed).sample(pairs, args.sample))

    # the images as imgfilter reads them, so that bit depth and color match what it filters
    work = REPO / "results" / f"tune_{random.Random().randrange(10 ** 8)}"
    try:
        work.mkdir(parents=True)
        din, dref = bench.link_pairs(work, pairs)
        args.nlm, args.canny, args.cold, args.trials, args.warmup = [], [], False, 1, 0
        nin = compare.images(bench.normalize(args, work, din, "input"))
        nref = compare.images(bench.normalize(args, work, dref, "reference"))
        keys = sorted(k for k in nin if k in nref and bench.image_size(nin[k]) == bench.image_size(nref[k]))
        if not keys:
            sys.exit("tune.py: no pair of images of the same size")
        refs = {k: bench.gray(nref[k]) for k in keys}
        ins = {k: bench.gray(nin[k]) for k in keys}
        indir = work / "norm" / "input"

        noisy_each = {k: compare.psnr(refs[k].astype(float), ins[k].astype(float)) for k in keys}
        noisy = mean(noisy_each)
        rms = float(np.mean([np.sqrt(np.mean((refs[k].astype(float) - ins[k].astype(float)) ** 2)) for k in keys]))
        truth_dir = work / "truth_edges"
        run_edges(args, work / "norm" / "reference", truth_dir)
        tfiles = compare.images(truth_dir)
        truth = {k: bench.gray(tfiles[k]) for k in keys}
        noisy_f1_each = f1s(args, truth, indir, work)
        noisy_f1 = mean(noisy_f1_each)
        print(f"{len(keys)} image pairs; noisy input: {noisy:.2f} dB, edge F1 {noisy_f1:.3f} "
              f"(RMS difference to the truth: {rms:.1f} gray levels)")

        out = work / "out"
        s0 = args.search[0]
        shutil.rmtree(out, ignore_errors=True)
        denoise(args, ["-S", str(s0)], indir, out, work / "auto.json")
        auto_each, auto_f1_each = psnrs(refs, out), f1s(args, truth, out, work)
        auto, auto_f1 = mean(auto_each), mean(auto_f1_each)
        j = json.load(open(work / "auto.json"))["denoise"]
        print(f"imgfilter's automatic setting: {auto:.2f} dB ({auto - noisy:+.2f}), edge F1 {auto_f1:.3f} "
              f"({auto_f1 - noisy_f1:+.3f}); noise {j['sigma_used']['mean']:.1f} gray levels, "
              f"strength {j['strength_used']['mean']:.2f}")

        grid = list(itertools.product(args.sigma, args.strength, args.patch, args.search))
        results = []
        for n, (sg, h, p, s) in enumerate(grid, 1):
            shutil.rmtree(out, ignore_errors=True)
            denoise(args, ["-N", sg, *([] if h == "auto" else ["-H", h]), "-P", str(p), "-S", str(s)], indir, out)
            ve, fe = psnrs(refs, out), f1s(args, truth, out, work)
            v, f = mean(ve), mean(fe)
            results.append((v, f, sg, h, p, s, ve, fe))
            print(f"[{n}/{len(grid)}] -N {sg} -H {h} -P {p} -S {s}: {v:.2f} dB, F1 {f:.3f}", file=sys.stderr, flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    col = 0 if args.metric == "psnr" else 1
    results.sort(key=lambda r: r[col], reverse=True)
    print(f"\nsorted by {'PSNR' if col == 0 else 'edge F1'}\n"
          f"{'':4s}{'-N':>8s}{'-H':>6s}{'-P':>4s}{'-S':>4s}{'PSNR dB':>10s}{'gain dB':>9s}{'F1':>8s}{'gain':>8s}")
    for i, (v, f, sg, h, p, s, *_) in enumerate(results[:args.top], 1):
        print(f"{i:3d} {sg:>8s}{h:>6s}{p:>4d}{s:>4d}{v:>10.2f}{v - noisy:>+9.2f}{f:>8.3f}{f - noisy_f1:>+8.3f}")

    def flags(r):
        return f"-N {r[2]} " + ("" if r[3] == "auto" else f"-H {r[3]} ") + f"-P {r[4]} -S {r[5]}"
    bp, bf = max(results, key=lambda r: r[0]), max(results, key=lambda r: r[1])
    print(f"\nautomatic setting: {auto:.2f} dB, F1 {auto_f1:.3f}")
    print(f"best PSNR: {bp[0]:.2f} dB, F1 {bp[1]:.3f}   imgfilter -d {flags(bp)} ...")
    print(f"best F1:   {bf[0]:.2f} dB, F1 {bf[1]:.3f}   imgfilter -d {flags(bf)} ...")
    print(f'use with bench.py: --nlm "{flags(results[0])}"')

    if args.tiers > 1 and len(keys) >= 2 * args.tiers:
        order = sorted(keys, key=lambda k: noisy_each[k])            # noisiest first
        print("\nby noise level (noisiest first): the best setting can differ between them")
        for t, tier in enumerate(np.array_split(order, args.tiers), 1):
            sub = lambda d: float(np.mean([d[k] for k in tier]))
            lo, hi = min(noisy_each[k] for k in tier), max(noisy_each[k] for k in tier)
            print(f"  tier {t}: {len(tier)} images, noisy PSNR {lo:.1f}-{hi:.1f} dB")
            rows = [("noisy input", None, sub(noisy_each), sub(noisy_f1_each)),
                    ("automatic", None, sub(auto_each), sub(auto_f1_each))]
            best_p = max(results, key=lambda r: sub(r[6]))
            best_f = max(results, key=lambda r: sub(r[7]))
            rows += [("best PSNR", flags(best_p), sub(best_p[6]), sub(best_p[7])),
                     ("best F1", flags(best_f), sub(best_f[6]), sub(best_f[7]))]
            for name, fl, pv, fv in rows:
                print(f"    {name:12s} {pv:6.2f} dB  F1 {fv:.3f}" + (f"   imgfilter -d {fl}" if fl else ""))
    edge = []   # parameters whose best value is the largest one tried
    for name, value, tried, key in (("--sigma", sg, args.sigma, lambda x: float(x.rstrip("xX"))),
                                    ("--strength", h, [t for t in args.strength if t != "auto"], float),
                                    ("--patch", p, args.patch, int)):
        if len(set(tried)) > 1 and value == max(tried, key=key):
            edge.append(name)
    if edge:
        print(f"note: the best setting is the largest value tried for {', '.join(edge)}; "
              "try larger values there and run again")

if __name__ == "__main__":
    main()
