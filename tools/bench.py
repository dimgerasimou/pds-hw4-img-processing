#!/usr/bin/env python3
"""
bench.py - Benchmarks and quality evaluation for imgfilter.

Steps (skip any with --skip):
  data     the input directory, or --sample N random images of it. The ground
           truth is either --reference DIR (files match by base name, no noise
           is added) or the input images themselves, with noise added (--noise)
  batch    batch-size sweep on the CPU and GPU, all threads
  threads  thread sweep on the CPU
  quality  PSNR and edge F1 of the noisy and denoised images against the truth,
           and a check that the CPU and GPU outputs are identical

Timed runs write no images unless --format is given. Results go to
results/<timestamp>/: runs/ (one JSON per imgfilter run), runs.csv, quality.csv,
summary.md, environment.txt, data/, and out/ with --keep-outputs. The quality
step works in chunks of --chunk images and deletes their files when done, so
memory and disk use stay bounded.

Examples:
  tools/bench.py --sample 200 data/chest_xray
  tools/bench.py --sample 200 --dose 0.2 --skip threads data/chest_xray
  tools/bench.py --reference data/aapm/reference --skip batch,threads data/aapm/input
"""

import argparse
import csv
import datetime
import json
import math
import os
import platform
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
    from PIL import Image
    import noise
except ImportError as e:
    sys.exit(f"bench.py: {e.name} is required (Arch: python-numpy python-pillow)")

FILTERS = {"denoise": ["-d"], "edges": ["-e"]}
STEPS = ("data", "batch", "threads", "quality")
THREAD_BATCH = 64


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def int_list(s):
    return [int(x) for x in s.replace(" ", ",").split(",") if x]


def default_threads():
    n, t, out = os.cpu_count() or 1, 1, []
    while t < n:
        out.append(t)
        t *= 2
    return out + [n]


def images(d):
    """Base name -> file, in file name order."""
    return {p.stem: p for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in noise.EXTENSIONS}


def gray(path):
    return np.array(Image.open(path).convert("L"))


def image_size(path):
    with Image.open(path) as im:
        return im.size


def psnr(ref, img):
    mse = np.mean((ref - img) ** 2)
    return math.inf if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)


def dilate(a):
    p = np.pad(a, 1)
    out = np.zeros_like(a)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out |= p[dy:dy + a.shape[0], dx:dx + a.shape[1]]
    return out


def edge_f1(ref, pred):
    """F1 of edge maps with 1-pixel tolerance."""
    r, p = ref > 0, pred > 0
    if not r.any() and not p.any():
        return 1.0
    prec = (p & dilate(r)).sum() / max(p.sum(), 1)
    rec = (r & dilate(p)).sum() / max(r.sum(), 1)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def identical(a, b):
    ia, ib = images(a), images(b)
    return ia.keys() == ib.keys() and all(ia[k].read_bytes() == ib[k].read_bytes() for k in ia)


def link(dst, files):
    dst.mkdir(parents=True)
    for name, f in files:
        (dst / name).symlink_to(f.resolve())


# ---------------------------------------------------------------- data

def prepare_data(args, res):
    """Returns (input dir for the runs, ground truth dir or None)."""
    ins = images(args.input)
    refs = images(args.reference) if args.reference else None
    keys = sorted(ins.keys() & refs.keys()) if refs else list(ins)
    if not keys:
        sys.exit("bench.py: " + ("no input image has a reference (file names must match)" if refs
                                 else f"no images in {args.input}"))
    if refs and len(keys) < max(len(ins), len(refs)):
        log("data: images without a partner are ignored")
    if args.sample:
        if args.sample > len(keys):
            sys.exit(f"bench.py: --sample {args.sample}, but only {len(keys)} images")
        keys = sorted(random.Random(args.seed).sample(keys, args.sample))

    if refs:
        link(res / "data" / "input", [(k + ins[k].suffix, ins[k]) for k in keys])
        link(res / "data" / "reference", [(k + refs[k].suffix, refs[k]) for k in keys])
        return res / "data" / "input", res / "data" / "reference"
    if args.noise == "none":
        if not args.sample:
            return args.input, None
        link(res / "data" / "input", [(ins[k].name, ins[k]) for k in keys])
        return res / "data" / "input", None

    # noise.py writes the clean copies as PNG too, so that every tool decodes the same pixels
    clean, noisy = res / "data" / "clean", res / "data" / "noisy"
    link(res / "data" / "originals", [(ins[k].name, ins[k]) for k in keys])
    log(f"data: {len(keys)} images, adding {args.noise} noise")
    cmd = [sys.executable, str(TOOLS / "noise.py"), "--model", args.noise, "--seed", str(args.seed),
           "--clean", str(clean)]
    cmd += ["--dose", str(args.dose)] if args.noise == "poisson" and args.dose is not None else []
    cmd += ["--sigma", str(args.sigma)] if args.noise == "gaussian" else []
    subprocess.run(cmd + [str(res / "data" / "originals"), str(noisy)], check=True, stdout=subprocess.DEVNULL)
    return noisy, clean


# ---------------------------------------------------------------- runs

def imgfilter(args, opts, inp, out=None, bench=None, timed=True):
    """Runs imgfilter; returns (exit code, stderr)."""
    cmd = [str(args.bin)] + opts + (args.nlm if "-d" in opts else [])
    if bench:
        cmd += ["-b", str(bench), "-n", str(args.trials if timed else 1)]
        if timed and args.warmup:
            cmd += ["-w", str(args.warmup)]
    if out:
        cmd += ["-o", str(out)]
    p = subprocess.run(cmd + [str(inp)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stderr


def gpu_available(args, inp):
    one = next(iter(images(inp).values()))
    code, err = imgfilter(args, ["-g", "-e", "-t", "1", "-G", "0"], one, timed=False)
    if code != 0:
        log(f"gpu: not available, skipping ({err.strip().splitlines()[-1] if err.strip() else code})")
    return code == 0


def row(sweep, device, filt, threads, batch, path):
    r = json.load(open(path))
    gpu = r.get("gpu_time") or {}
    stage = r["results"].get(filt)
    first = r["results"]["read"]
    return {
        "sweep": sweep, "device": device, "filter": filt, "threads": threads, "batch": batch,
        "pipeline_s": r["pipeline_time"]["median_time_s"],
        "pipeline_min_s": r["pipeline_time"]["min_time_s"],
        "filter_s": stage["wall_time"]["median_time_s"] if stage else None,
        "gpu_kernels_s": sum(gpu[k]["median_time_s"] for k in ("denoise_kernel", "edges_kernels") if k in gpu) if gpu else None,
        "peak_rss_gb": r["memory"]["cpu_peak_rss_gb"],
        "images": first["images"] if first else None,
        "json": path.name,
    }


def sweep(args, res, inp, name, configs):
    """configs: (device, filter, threads, batch) tuples. Returns the result rows."""
    rows, out = [], None
    (res / "runs").mkdir(exist_ok=True)
    for i, (dev, filt, t, b) in enumerate(configs, 1):
        path = res / "runs" / f"{name}_{dev}_{filt}_t{t}_B{b}.json"
        log(f"[{name} {i}/{len(configs)}] {dev} {filt} threads={t} batch={b}")
        opts = FILTERS[filt] + ["-t", str(t), "-B", str(b)] + (["-g"] if dev == "gpu" else [])
        if args.format:
            out = res / "scratch"
            shutil.rmtree(out, ignore_errors=True)
            opts += ["-f", args.format]
        code, err = imgfilter(args, opts, inp, out=out, bench=path)
        if code != 0:
            log(f"  imgfilter exited with {code}: {err.strip()[-300:]}")
        if path.exists():
            rows.append(row(name, dev, filt, t, b, path))
    shutil.rmtree(res / "scratch", ignore_errors=True)
    return rows


# ---------------------------------------------------------------- quality

def normalize(args, work, src, name):
    """The images as imgfilter reads them (8-bit gray): truth and input go through the same
    conversion the filters saw; Pillow would clip 16-bit images instead of scaling them."""
    d = work / "norm" / name
    d.parent.mkdir(exist_ok=True)   # imgfilter creates only the last directory
    code, err = imgfilter(args, ["-B", "16"], src, out=d, timed=False)
    if code != 0:
        sys.exit(f"bench.py: cannot read the {name} images ({code}): {err.strip()[-300:]}")
    return d


def quality_chunk(args, res, work, noisy, clean, gpu, tag):
    """PSNR and edge F1 of one chunk of image pairs; the files are kept in work/."""
    out = work / "out"
    out.mkdir(parents=True)
    runs = {"denoised": (["-d"], noisy), "edges_clean": (["-e"], clean),
            "edges_noisy": (["-e"], noisy), "edges_denoised": (["-d", "-e"], noisy)}
    if gpu:
        runs["denoised_gpu"] = (["-d", "-g"], noisy)
        runs["edges_denoised_gpu"] = (["-d", "-e", "-g"], noisy)

    used = {}
    for name, (opts, inp) in runs.items():
        log(f"[quality{tag}] {name}")
        bench = res / "runs" / f"quality_{name}{tag.replace(' ', '_').replace('/', 'of')}.json"
        code, err = imgfilter(args, opts + ["-B", "16"], inp, out=out / name, bench=bench, timed=False)
        if code != 0:
            sys.exit(f"bench.py: quality run {name!r} failed ({code}): {err.strip()[-300:]}")
        if name == "denoised":
            j = json.load(open(bench))["denoise"]
            used = {"sigma": j["sigma_used"]["mean"], "strength": j.get("strength_used", {}).get("mean")}

    ref, nz = images(normalize(args, work, clean, "reference")), images(normalize(args, work, noisy, "input"))
    dn = images(out / "denoised")
    ec, en, ed = (images(out / d) for d in ("edges_clean", "edges_noisy", "edges_denoised"))
    keys = sorted(set(ref) & set(nz) & set(dn) & set(ec) & set(en) & set(ed))
    odd = [k for k in keys if image_size(ref[k]) != image_size(nz[k])]
    if odd:
        log(f"quality: {len(odd)} pairs skipped, input and reference differ in size: "
            + ", ".join(odd[:5]) + (", ..." if len(odd) > 5 else ""))

    rows = []
    for k in (k for k in keys if k not in odd):   # one image at a time
        r, n, d = (gray(p[k]).astype(float) for p in (ref, nz, dn))
        e = gray(ec[k])
        rows.append({"image": k, "psnr_noisy": psnr(r, n), "psnr_denoised": psnr(r, d),
                     "edge_f1_noisy": edge_f1(e, gray(en[k])), "edge_f1_denoised": edge_f1(e, gray(ed[k]))})

    checks = {}
    if gpu:
        checks["denoise: CPU and GPU identical"] = identical(out / "denoised", out / "denoised_gpu")
        checks["denoise + edges: CPU and GPU identical"] = identical(out / "edges_denoised", out / "edges_denoised_gpu")
    return rows, used, checks


def weighted(used, key):
    """Mean of a per-chunk value, weighted by the chunk sizes."""
    v = [(n, u[key]) for n, u in used if u.get(key)]
    return sum(n * x for n, x in v) / sum(n for n, _ in v) if v else None


def quality(args, res, noisy, clean, gpu):
    """The quality step in chunks of --chunk pairs. Each chunk's files are deleted when done,
    or moved to out/ with --keep-outputs."""
    inp, ref = images(noisy), images(clean)
    keys = sorted(inp.keys() & ref.keys())
    if not keys:
        sys.exit("bench.py: no input image has a reference for the quality step")
    chunks = [keys[i:i + args.chunk] for i in range(0, len(keys), args.chunk)]

    rows, checks, used = [], {}, []
    work = res / "work"
    for c, chunk in enumerate(chunks, 1):
        shutil.rmtree(work, ignore_errors=True)
        link(work / "input", [(inp[k].name, inp[k]) for k in chunk])
        link(work / "reference", [(ref[k].name, ref[k]) for k in chunk])
        r, u, chk = quality_chunk(args, res, work, work / "input", work / "reference", gpu,
                                  f" {c}/{len(chunks)}" if len(chunks) > 1 else "")
        rows += r
        used.append((len(chunk), u))
        for name, ok in chk.items():
            checks[name] = checks.get(name, True) and ok
        if args.keep_outputs:
            for d in (work / "out").iterdir():
                (res / "out" / d.name).mkdir(parents=True, exist_ok=True)
                for f in d.iterdir():
                    os.replace(f, res / "out" / d.name / f.name)
        shutil.rmtree(work)

    if not rows:
        sys.exit("bench.py: no image pair left for the quality step")

    added = None
    if (noisy / "noise.csv").exists():
        manifest = {r["file"].rsplit(".", 1)[0]: float(r["noise_std"]) for r in csv.DictReader(open(noisy / "noise.csv"))}
        got = [manifest[r["image"]] for r in rows if r["image"] in manifest]
        added = sum(got) / len(got) if got else None
    return rows, {"added": added, "used": weighted(used, "sigma"), "strength": weighted(used, "strength")}, \
        checks, len(ref) - len(rows)


# ---------------------------------------------------------------- report

def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)


def fmt(v, spec=".3f"):
    return "-" if v is None else format(v, spec)


def times(v):
    return "-" if v is None else f"{v:.2f}x"


def table(head, body):
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---:" if i else "---" for i in range(len(head))) + "|"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in body])


def finite_mean(values):
    v = [x for x in values if math.isfinite(x)]
    return sum(v) / len(v) if v else math.inf


def summary(batch_rows, thread_rows, qual, noise_lv, checks, filters, skipped=0):
    md = []
    for filt in filters:
        b = [r for r in batch_rows if r["filter"] == filt]
        if b:
            md.append(f"## Batch sweep: {filt}\n")
            body = []
            for bs in sorted({r["batch"] for r in b}):
                c = next((r for r in b if r["batch"] == bs and r["device"] == "cpu"), None)
                g = next((r for r in b if r["batch"] == bs and r["device"] == "gpu"), None)
                body.append([bs or "all", fmt(c and c["pipeline_s"]), fmt(c and c["filter_s"]), fmt(c and c["peak_rss_gb"], ".2f"),
                             fmt(g and g["pipeline_s"]), fmt(g and g["gpu_kernels_s"]), fmt(g and g["peak_rss_gb"], ".2f"),
                             times(c["pipeline_s"] / g["pipeline_s"] if c and g else None)])
            md.append(table(["batch", "CPU pipeline s", "CPU filter s", "CPU peak GB", "GPU pipeline s",
                             "GPU kernels s (sum)", "GPU peak GB", "GPU speedup"], body) + "\n")
            md.append("GPU kernels: sum of the per-image kernel times; images overlap on the GPU, "
                      "so the sum can exceed the pipeline time.\n")
        t = sorted((r for r in thread_rows if r["filter"] == filt), key=lambda r: r["threads"])
        if t:
            md.append(f"## Thread sweep (CPU): {filt}\n")
            base = next((r for r in t if r["threads"] == 1), None)
            body = []
            for r in t:
                sp = base["pipeline_s"] / r["pipeline_s"] if base else None
                spf = base["filter_s"] / r["filter_s"] if base and base["filter_s"] and r["filter_s"] else None
                body.append([r["threads"], fmt(r["pipeline_s"]), fmt(r["filter_s"]), times(sp), times(spf),
                             fmt(sp / r["threads"] if sp else None, ".0%")])
            md.append(table(["threads", "pipeline s", "filter s", "speedup", "filter speedup", "efficiency"], body) + "\n")
        best = {}
        for r in batch_rows + thread_rows:
            if r["filter"] == filt and (r["device"] not in best or r["pipeline_s"] < best[r["device"]]["pipeline_s"]):
                best[r["device"]] = r
        if "cpu" in best and "gpu" in best:
            c, g = best["cpu"], best["gpu"]
            md.append(f"**CPU vs GPU ({filt}), best configuration of each:** CPU {c['pipeline_s']:.3f} s "
                      f"(t={c['threads']}, B={c['batch'] or 'all'}), GPU {g['pipeline_s']:.3f} s "
                      f"(B={g['batch'] or 'all'}): **{c['pipeline_s'] / g['pipeline_s']:.2f}x**\n")
            serial = next((r for r in thread_rows if r["filter"] == filt and r["threads"] == 1), None)
            if serial:
                md.append(f"**GPU vs serial CPU (1 thread, {filt}):** **{serial['pipeline_s'] / g['pipeline_s']:.2f}x**\n")
    if qual:
        mean = lambda k: finite_mean([r[k] for r in qual])
        worst = lambda k: min(r[k] for r in qual)
        md.append(f"## Quality ({len(qual)} images" + (f", {skipped} skipped: different sizes" if skipped else "") + ")\n")
        md.append(table(["", "noisy", "denoised"], [
            ["PSNR mean dB", f"{mean('psnr_noisy'):.2f}", f"{mean('psnr_denoised'):.2f}"],
            ["PSNR worst dB", f"{worst('psnr_noisy'):.2f}", f"{worst('psnr_denoised'):.2f}"],
            ["edge F1 mean", f"{mean('edge_f1_noisy'):.3f}", f"{mean('edge_f1_denoised'):.3f}"],
            ["edge F1 worst", f"{worst('edge_f1_noisy'):.3f}", f"{worst('edge_f1_denoised'):.3f}"],
        ]) + "\n")
        md.append("Edge F1: edges of the noisy / denoised images against those of the clean images, 1-pixel tolerance.\n")
        if noise_lv.get("used") is not None:
            line = (f"Noise level: added {noise_lv['added']:.2f}, used by imgfilter {noise_lv['used']:.2f}"
                    if noise_lv.get("added") is not None else f"Noise level used by imgfilter: {noise_lv['used']:.2f}")
            line += " (mean std, gray levels)"
            if noise_lv.get("strength"):
                line += f"; strength h/sigma {noise_lv['strength']:.2f} (milder in the edge runs for near-white noise)"
            md.append(line + ".\n")
        md.extend(f"- {k}: **{'yes' if v else 'NO'}**" for k, v in checks.items())
        md.append("")
    return "\n".join(md)


def describe_data(args, n):
    if args.reference:
        return f"real pairs: {n} images of {args.input} with the truth in {args.reference}" + \
               (f" (sampled, seed {args.seed})" if args.sample else "")
    if args.noise == "none":
        return f"{n} images of {args.input}, no ground truth"
    return f"{n} images of {args.input} with added {args.noise} noise (seed {args.seed}); the originals are the truth"


def environment(args, res, gpu, data):
    def run(c):
        try:
            return subprocess.run(c, capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
        except (OSError, subprocess.TimeoutExpired):
            return "unknown"
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "unknown")
    try:
        gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    except OSError:
        gov = "unknown"
    commit = run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"])
    if commit != "unknown" and subprocess.run(["git", "-C", str(REPO), "diff", "--quiet"]).returncode != 0:
        commit += " (dirty)"
    lines = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"), "host": platform.node(), "cpu": cpu,
        "nproc": os.cpu_count(),
        "gpu": run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]) if gpu else "none",
        "commit": commit, "command": " ".join(sys.argv), "data": data, "governor": gov,
        "OMP_PROC_BIND": os.environ["OMP_PROC_BIND"], "OMP_PLACES": os.environ["OMP_PLACES"],
        "nlm options": " ".join(args.nlm) or "defaults",
    }
    (res / "environment.txt").write_text("".join(f"{k + ':':15s}{v}\n" for k, v in lines.items()))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Benchmarks and quality evaluation for imgfilter.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("input", type=Path, help="directory of images")
    ap.add_argument("--sample", type=int, help="use N random images of the input")
    ap.add_argument("--seed", type=int, default=0, help="for sampling and noise (default: 0)")
    ap.add_argument("--reference", type=Path, help="directory with the ground truth (no noise is added)")
    ap.add_argument("--noise", choices=("poisson", "gaussian", "none"), help="noise to add (default: poisson)")
    ap.add_argument("--dose", type=float, help="poisson: fixed dose (default: random per image, 10-50%%)")
    ap.add_argument("--sigma", type=float, default=20.0, help="gaussian: noise std (default: 20)")
    ap.add_argument("--filters", default="denoise,edges", help="denoise, edges (default: both)")
    ap.add_argument("--devices", default="cpu,gpu", help="for the batch sweep (default: cpu,gpu)")
    ap.add_argument("--batches", type=int_list, default=[4, 16, 64, 256], help="batch sizes, 0 = all (default: 4,16,64,256)")
    ap.add_argument("--threads", type=int_list, default=default_threads(),
                    help="thread counts (default: powers of 2 up to nproc, and nproc)")
    ap.add_argument("-n", "--trials", type=int, default=3, help="timed trials per run (default: 3)")
    ap.add_argument("-w", "--warmup", type=int, default=1, help="warmup trials per run (default: 1)")
    ap.add_argument("--format", choices=("pgm", "png"), help="also write images in timed runs")
    ap.add_argument("--nlm", default="", help='NLM options for denoising runs, e.g. "-S 7 -H 0.5"')
    ap.add_argument("--skip", default="", help="steps to skip: " + ", ".join(STEPS))
    ap.add_argument("--chunk", type=int, default=500, help="images per chunk of the quality step (default: 500)")
    ap.add_argument("--keep-outputs", action="store_true", help="keep the quality step's images in out/ (11 MB each)")
    ap.add_argument("--bin", type=Path, default=REPO / "bin" / "imgfilter")
    ap.add_argument("--no-build", action="store_true", help="do not run make first")
    ap.add_argument("-o", "--results", type=Path, help="results directory (default: results/<timestamp>)")
    args = ap.parse_args()

    args.nlm = args.nlm.split()
    filters = [f for f in args.filters.split(",") if f]
    devices = [d for d in args.devices.split(",") if d]
    skip = {s for s in args.skip.split(",") if s}
    if not set(filters) <= set(FILTERS):
        ap.error("--filters: denoise, edges")
    if not skip <= set(STEPS):
        ap.error(f"unknown step in --skip (steps: {', '.join(STEPS)})")
    if not args.input.is_dir() or (args.reference and not args.reference.is_dir()):
        ap.error("the input and --reference must be directories")
    if args.reference and args.noise not in (None, "none"):
        ap.error("--reference brings its own truth: no noise is added")
    args.noise = "none" if args.reference else args.noise or "poisson"
    if args.chunk < 1:
        ap.error("--chunk must be at least 1")

    # threads on cores first, then their second hardware threads
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")
    os.environ.setdefault("OMP_DYNAMIC", "false")

    if not args.no_build:
        subprocess.run(["make", "-s", "-C", str(REPO)], check=True, stdout=subprocess.DEVNULL)

    res = args.results or REPO / "results" / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    res.mkdir(parents=True, exist_ok=False)
    (res / "runs").mkdir()

    if "data" not in skip:
        try:
            inp, clean = prepare_data(args, res)
        except SystemExit:
            shutil.rmtree(res, ignore_errors=True)   # nothing was measured yet
            raise
    else:
        inp, clean = args.input, args.reference   # names must match already
    data = describe_data(args, len(images(inp)))
    gpu = "gpu" in devices and gpu_available(args, inp)
    devices = [d for d in devices if d != "gpu" or gpu]
    environment(args, res, gpu, data)

    nmax = os.cpu_count() or 1
    nb = 0 if "batch" in skip else len(filters) * len(devices) * len(args.batches)
    nt = 0 if "threads" in skip else len(filters) * len(args.threads)
    nchunk = -(-len(images(inp)) // args.chunk)
    nq = 0 if "quality" in skip or clean is None else ((6 if gpu else 4) + 2) * nchunk   # + reading the images
    log(f"plan: {nb} batch-sweep runs and {nt} thread-sweep runs ({args.trials} trials"
        + (f" + {args.warmup} warmup" if args.warmup else "") + f" each), {nq} quality runs" + (f" in {nchunk} chunks" if nq else ""))

    batch_rows = thread_rows = []
    if "batch" not in skip:
        batch_rows = sweep(args, res, inp, "batch", [(d, f, nmax, b) for f in filters for d in devices for b in args.batches])
    if "threads" not in skip:
        thread_rows = sweep(args, res, inp, "threads", [("cpu", f, t, THREAD_BATCH) for f in filters for t in args.threads])
    write_csv(res / "runs.csv", batch_rows + thread_rows)

    qual, noise_lv, checks, skipped = [], {}, {}, 0
    if "quality" not in skip:
        if clean is None:
            log("quality: skipped, needs a ground truth (noise added to the images, or --reference)")
        else:
            qual, noise_lv, checks, skipped = quality(args, res, inp, clean, gpu)
            write_csv(res / "quality.csv", qual)

    md = summary(batch_rows, thread_rows, qual, noise_lv, checks, filters, skipped)
    (res / "summary.md").write_text(f"# imgfilter benchmark, {res.name}\n\nData: {data}\n\n{md}")
    print(md)
    log(f"results in {res}")


if __name__ == "__main__":
    main()
