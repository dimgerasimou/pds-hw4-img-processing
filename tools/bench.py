#!/usr/bin/env python3
"""
bench.py - Benchmarks and quality evaluation for imgfilter.

Steps (skip any with --skip):
  data     the input directory as it is, or --sample N random images of it;
           with noise (the default), clean PNG references and noisy copies
  batch    batch-size sweep on every device, all threads
  threads  thread sweep on the CPU, fixed batch size
  quality  PSNR of noisy and denoised images against the clean ones, edge
           accuracy against the edges of the clean images, and a check that
           the CPU and GPU produce identical outputs

Timed runs write no images unless --format is given, so that disk writes do
not dominate the timings. Results go to one directory: runs/ (one JSON per
imgfilter run), runs.csv, quality.csv, summary.md, environment.txt, and the
generated data/ and out/.

Requires numpy and Pillow (Arch: python-numpy python-pillow).

Examples:
  tools/bench.py --sample 200 data/chest_xray
  tools/bench.py --sample 200 --dose 0.2 --skip threads data/chest_xray
  tools/bench.py --noise none --filters edges --batches 16,64 data/chest_xray
"""

import argparse
import csv
import datetime
import json
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
sys.dont_write_bytecode = True  # no __pycache__ in tools/ from the imports below

try:
    import numpy as np
    from PIL import Image
    import compare
    import noise
except ImportError as e:
    sys.exit(f"bench.py: {e.name} is required (Arch: python-numpy python-pillow)")

FILTERS = {"denoise": ["-d"], "edges": ["-e"], "both": ["-d", "-e"]}
STEPS = ("data", "batch", "threads", "quality")


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


# ---------------------------------------------------------------- data

def prepare_data(args, res):
    """Returns (input dir for the runs, clean reference dir or None)."""
    files = noise.list_inputs(args.input)
    if args.sample:
        if args.sample > len(files):
            sys.exit(f"bench.py: --sample {args.sample}, but only {len(files)} images")
        files = sorted(random.Random(args.seed).sample(files, args.sample))

    if args.noise == "none":
        if not args.sample:
            return args.input, None
        # a subset without noise: link the chosen originals
        d = res / "data" / "input"
        d.mkdir(parents=True)
        for f in files:
            (d / f.name).symlink_to(f.resolve())
        return d, None

    # clean references as PNG, so that every tool decodes the same pixels
    clean, noisy = res / "data" / "clean", res / "data" / "noisy"
    clean.mkdir(parents=True)
    log(f"data: {len(files)} images -> {clean}")
    for f in files:
        Image.fromarray(noise.load_gray(f)).save(clean / (f.stem + ".png"), compress_level=1)

    cmd = [sys.executable, str(TOOLS / "noise.py"), "--model", args.noise,
           "--seed", str(args.seed), str(clean), str(noisy)]
    if args.noise == "poisson":
        if args.dose is not None:
            cmd += ["--dose", str(args.dose)]
        cmd += ["--min-dose", str(args.min_dose), "--max-dose", str(args.max_dose)]
    else:
        cmd += ["--sigma", str(args.sigma)]
    log(f"data: adding {args.noise} noise -> {noisy}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return noisy, clean


# ---------------------------------------------------------------- runs

def drop_caches():
    subprocess.run(["sync"], check=True)
    subprocess.run(["sudo", "tee", "/proc/sys/vm/drop_caches"], input=b"3",
                   stdout=subprocess.DEVNULL, check=True)


def imgfilter(args, opts, inp, out=None, bench=None, timed=True):
    """Runs imgfilter; returns (exit code, stderr). Filter options go with their filter."""
    cmd = [str(args.bin)] + opts
    if "-d" in opts:
        cmd += args.nlm
    if "-e" in opts:
        cmd += args.canny
    if bench:
        cmd += ["-b", str(bench), "-n", str(args.trials if timed else 1)]
        if timed and args.warmup and not args.cold:
            cmd += ["-w", str(args.warmup)]
    if out:
        cmd += ["-o", str(out)]
    if args.cold and timed:
        drop_caches()
    p = subprocess.run(cmd + [str(inp)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stderr


def gpu_available(args, inp):
    one = next(iter(sorted(compare.images(inp).values())))
    code, err = imgfilter(args, ["-g", "-e", "-t", "1", "-G", "0"], one, timed=False)
    if code != 0:
        log(f"gpu: not available, skipping ({err.strip().splitlines()[-1] if err.strip() else code})")
    return code == 0


def stage_time(r, filt):
    """Median wall time of the filter's stages, or None."""
    stages = {"denoise": ["denoise"], "edges": ["edges"], "both": ["denoise", "edges"]}[filt]
    t = [r["results"][s]["wall_time"]["median_time_s"] for s in stages if r["results"].get(s)]
    return sum(t) if t else None


def row(sweep, device, filt, threads, batch, path):
    r = json.load(open(path))
    gpu = r.get("gpu_time") or {}
    kern = sum(gpu[k]["median_time_s"] for k in ("denoise_kernel", "edges_kernels") if k in gpu)
    first = r["results"]["read"]
    return {
        "sweep": sweep, "device": device, "filter": filt, "threads": threads, "batch": batch,
        "pipeline_s": r["pipeline_time"]["median_time_s"],
        "pipeline_min_s": r["pipeline_time"]["min_time_s"],
        "filter_s": stage_time(r, filt),
        "gpu_kernels_s": kern if gpu else None,
        "peak_rss_gb": r["memory"]["cpu_peak_rss_gb"],
        "images": first["images"] if first else None,
        "json": str(path.name),
    }


def sweep(args, res, inp, name, configs):
    """configs: list of (device, filter, threads, batch). Returns result rows."""
    rows, out = [], None
    d = res / "runs"
    d.mkdir(exist_ok=True)
    for i, (dev, filt, t, b) in enumerate(configs, 1):
        tag = f"{name}_{dev}_{filt}_t{t}_B{b}"
        log(f"[{name} {i}/{len(configs)}] {dev} {filt} threads={t} batch={b}")
        opts = FILTERS[filt] + ["-t", str(t), "-B", str(b)] + (["-g"] if dev == "gpu" else [])
        if args.format:
            out = res / "scratch"
            shutil.rmtree(out, ignore_errors=True)
            opts += ["-f", args.format]
        code, err = imgfilter(args, opts, inp, out=out, bench=d / f"{tag}.json")
        if code != 0:
            log(f"  imgfilter exited with {code}: {err.strip()[-300:]}")
        if (d / f"{tag}.json").exists():
            rows.append(row(name, dev, filt, t, b, d / f"{tag}.json"))
    shutil.rmtree(res / "scratch", ignore_errors=True)
    return rows


# ---------------------------------------------------------------- quality

def load_dir(d):
    return {stem: np.array(Image.open(p).convert("L")) for stem, p in compare.images(d).items()}


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
    ia, ib = compare.images(a), compare.images(b)
    return ia.keys() == ib.keys() and all(
        Path(ia[k]).read_bytes() == Path(ib[k]).read_bytes() for k in ia)


def quality(args, res, noisy, clean, gpu):
    out = res / "out"
    out.mkdir(exist_ok=True)
    runs = {  # name: (options, input)
        "denoised": (["-d"], noisy),
        "edges_clean": (["-e"], clean),
        "edges_noisy": (["-e"], noisy),
        "edges_denoised": (["-d", "-e"], noisy),
    }
    if gpu:
        runs["denoised_gpu"] = (["-d", "-g"], noisy)
        runs["edges_denoised_gpu"] = (["-d", "-e", "-g"], noisy)

    sig = {}
    for name, (opts, inp) in runs.items():
        log(f"[quality] {name}")
        code, err = imgfilter(args, opts + ["-B", "16"], inp, out=out / name,
                              bench=res / "runs" / f"quality_{name}.json", timed=False)
        if code != 0:
            sys.exit(f"bench.py: quality run {name!r} failed ({code}): {err.strip()[-300:]}")
        if name == "denoised":
            sig = json.load(open(res / "runs" / f"quality_{name}.json"))["denoise"]["sigma_used"]

    ref, nz, dn = load_dir(clean), load_dir(noisy), load_dir(out / "denoised")
    ec, en, ed = load_dir(out / "edges_clean"), load_dir(out / "edges_noisy"), load_dir(out / "edges_denoised")
    rows = []
    for k in sorted(ref):
        rows.append({
            "image": k,
            "psnr_noisy": compare.psnr(ref[k].astype(float), nz[k].astype(float)),
            "psnr_denoised": compare.psnr(ref[k].astype(float), dn[k].astype(float)),
            "edge_f1_noisy": edge_f1(ec[k], en[k]),
            "edge_f1_denoised": edge_f1(ec[k], ed[k]),
        })

    manifest = {r["file"].rsplit(".", 1)[0]: float(r["noise_std"])
                for r in csv.DictReader(open(noisy / "noise.csv"))}
    checks = {}
    if gpu:
        checks["denoise: CPU and GPU identical"] = identical(out / "denoised", out / "denoised_gpu")
        checks["denoise + edges: CPU and GPU identical"] = identical(out / "edges_denoised",
                                                                    out / "edges_denoised_gpu")
    return rows, {"added": sum(manifest.values()) / len(manifest), "estimated": sig.get("mean")}, checks


# ---------------------------------------------------------------- report

def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)


def fmt(v, spec=".3f"):
    return "-" if v is None else format(v, spec)


def table(head, body):
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---:" if i else "---" for i in range(len(head))) + "|"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in body])


def summary(batch_rows, thread_rows, qual, noise_lv, checks, filters):
    md = []
    for filt in filters:
        b = [r for r in batch_rows if r["filter"] == filt]
        if b:
            md.append(f"## Batch sweep: {filt}\n")
            body = []
            for bs in sorted({r["batch"] for r in b}):
                c = next((r for r in b if r["batch"] == bs and r["device"] == "cpu"), None)
                g = next((r for r in b if r["batch"] == bs and r["device"] == "gpu"), None)
                sp = c["pipeline_s"] / g["pipeline_s"] if c and g else None
                body.append([bs or "all", fmt(c and c["pipeline_s"]), fmt(c and c["filter_s"]),
                             fmt(c and c["peak_rss_gb"], ".2f"), fmt(g and g["pipeline_s"]),
                             fmt(g and g["gpu_kernels_s"]), fmt(g and g["peak_rss_gb"], ".2f"),
                             fmt(sp, ".2f") + ("x" if sp else "")])
            md.append(table(["batch", "CPU pipeline s", "CPU filter s", "CPU peak GB",
                             "GPU pipeline s", "GPU kernels s (sum)", "GPU peak GB", "GPU speedup"], body) + "\n")
            md.append("CPU filter: wall time of the filter stage. GPU kernels: sum of the per-image "
                      "kernel times measured by the GPU; consecutive images run concurrently on the "
                      "GPU, so the sum can exceed the pipeline time.\n")
        t = sorted((r for r in thread_rows if r["filter"] == filt), key=lambda r: r["threads"])
        if t:
            md.append(f"## Thread sweep (CPU): {filt}\n")
            base = next((r for r in t if r["threads"] == 1), None)
            body = []
            for r in t:
                sp = base["pipeline_s"] / r["pipeline_s"] if base else None
                spf = base["filter_s"] / r["filter_s"] if base and base["filter_s"] and r["filter_s"] else None
                body.append([r["threads"], fmt(r["pipeline_s"]), fmt(r["filter_s"]),
                             fmt(sp, ".2f") + ("x" if sp else ""), fmt(spf, ".2f") + ("x" if spf else ""),
                             fmt(sp / r["threads"] if sp else None, ".0%")])
            md.append(table(["threads", "pipeline s", "filter s", "speedup", "filter speedup",
                             "efficiency"], body) + "\n")
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
                md.append(f"**GPU vs serial CPU (1 thread, {filt}):** "
                          f"**{serial['pipeline_s'] / g['pipeline_s']:.2f}x**\n")
    if qual:
        n = len(qual)
        mean = lambda k: sum(r[k] for r in qual) / n
        worst = lambda k: min(r[k] for r in qual)
        md.append(f"## Quality ({n} images)\n")
        md.append(table(["", "noisy", "denoised"], [
            ["PSNR mean dB", f"{mean('psnr_noisy'):.2f}", f"{mean('psnr_denoised'):.2f}"],
            ["PSNR worst dB", f"{worst('psnr_noisy'):.2f}", f"{worst('psnr_denoised'):.2f}"],
            ["edge F1 mean", f"{mean('edge_f1_noisy'):.3f}", f"{mean('edge_f1_denoised'):.3f}"],
            ["edge F1 worst", f"{worst('edge_f1_noisy'):.3f}", f"{worst('edge_f1_denoised'):.3f}"],
        ]) + "\n")
        md.append("Edge F1: edges found on the noisy / denoised images against the edges of the "
                  "clean images, 1-pixel tolerance.\n")
        if noise_lv.get("estimated") is not None:
            md.append(f"Noise level: added {noise_lv['added']:.2f}, estimated by imgfilter "
                      f"{noise_lv['estimated']:.2f} (mean std, gray levels).\n")
        for k, v in checks.items():
            md.append(f"- {k}: **{'yes' if v else 'NO'}**")
        md.append("")
    return "\n".join(md)


def environment(args, res, gpu):
    def cmd(c):
        try:
            return subprocess.run(c, capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
        except (OSError, subprocess.TimeoutExpired):
            return "unknown"
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "unknown")
    try:
        gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    except OSError:
        gov = "unknown"
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True)
    commit = git.stdout.strip() if git.returncode == 0 else "unknown"
    if git.returncode == 0 and subprocess.run(["git", "-C", str(REPO), "diff", "--quiet"],
                                              capture_output=True).returncode != 0:
        commit += " (dirty)"
    lines = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(), "kernel": platform.release(), "cpu": cpu, "nproc": os.cpu_count(),
        "gpu": cmd(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]) if gpu else "none",
        "commit": commit,
        "command": " ".join(sys.argv),
        "governor": gov, "cache": "cold" if args.cold else "warm",
        "OMP_PROC_BIND": os.environ["OMP_PROC_BIND"], "OMP_PLACES": os.environ["OMP_PLACES"],
        "nlm options": " ".join(args.nlm) or "defaults",
        "canny options": " ".join(args.canny) or "defaults",
        "note": args.note or "",
    }
    (res / "environment.txt").write_text("".join(f"{k + ':':15s}{v}\n" for k, v in lines.items()))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Benchmarks and quality evaluation for imgfilter.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("input", type=Path, help="directory of images")
    ap.add_argument("--sample", type=int, help="use N random images of the input")
    ap.add_argument("--seed", type=int, default=0, help="for sampling and noise (default: 0)")
    ap.add_argument("--noise", choices=("poisson", "gaussian", "none"), default="poisson",
                    help="noise added to the images (default: poisson)")
    ap.add_argument("--dose", type=float, help="poisson: fixed dose (default: random per image)")
    ap.add_argument("--min-dose", type=float, default=0.10)
    ap.add_argument("--max-dose", type=float, default=0.50)
    ap.add_argument("--sigma", type=float, default=20.0, help="gaussian: noise std (default: 20)")
    ap.add_argument("--filters", default="denoise,edges", help="denoise, edges, both (default: denoise,edges)")
    ap.add_argument("--devices", default="cpu,gpu", help="for the batch sweep (default: cpu,gpu)")
    ap.add_argument("--batches", type=int_list, default=[4, 16, 64, 256],
                    help="batch sizes, 0 = all (default: 4,16,64,256)")
    ap.add_argument("--threads", type=int_list, default=default_threads(),
                    help="thread counts (default: powers of 2 up to nproc, and nproc)")
    ap.add_argument("--thread-batch", type=int, default=64, help="batch size of the thread sweep (default: 64)")
    ap.add_argument("-n", "--trials", type=int, default=3, help="timed trials per run (default: 3)")
    ap.add_argument("-w", "--warmup", type=int, default=1, help="warmup trials per run (default: 1)")
    ap.add_argument("--format", choices=("pgm", "png"), help="also write images in timed runs")
    ap.add_argument("--nlm", default="", help='NLM options for denoising runs, e.g. "-S 7 -H 0.5"')
    ap.add_argument("--canny", default="", help='Canny options for edge runs, e.g. "-G 2 -l 10 -u 40"')
    ap.add_argument("--cold", action="store_true", help="drop the page cache before every timed run (sudo)")
    ap.add_argument("--skip", default="", help="steps to skip: " + ", ".join(STEPS))
    ap.add_argument("--bin", type=Path, default=REPO / "bin" / "imgfilter")
    ap.add_argument("--no-build", action="store_true", help="do not run make first")
    ap.add_argument("--note", help="free text for environment.txt, e.g. the power profile")
    ap.add_argument("-o", "--results", type=Path, help="results directory (default: results/<timestamp>)")
    args = ap.parse_args()

    args.nlm, args.canny = args.nlm.split(), args.canny.split()
    filters = [f for f in args.filters.split(",") if f]
    devices = [d for d in args.devices.split(",") if d]
    skip = {s for s in args.skip.split(",") if s}
    for f in filters:
        if f not in FILTERS:
            ap.error(f"unknown filter {f!r} (denoise, edges, both)")
    if not skip <= set(STEPS):
        ap.error(f"unknown step in --skip (steps: {', '.join(STEPS)})")
    if not args.input.is_dir():
        ap.error(f"{args.input} is not a directory")
    if args.cold:
        subprocess.run(["sudo", "-v"], check=True)

    # threads on cores first, then their second hardware threads
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")
    os.environ.setdefault("OMP_DYNAMIC", "false")

    if not args.no_build:
        subprocess.run(["make", "-s", "-C", str(REPO)], check=True, stdout=subprocess.DEVNULL)

    res = args.results or REPO / "results" / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    res.mkdir(parents=True, exist_ok=False)
    (res / "runs").mkdir()

    inp, clean = prepare_data(args, res) if "data" not in skip else (args.input, None)
    gpu = "gpu" in devices and gpu_available(args, inp)
    if not gpu:
        devices = [d for d in devices if d != "gpu"]
    environment(args, res, gpu)

    nmax = os.cpu_count() or 1
    nb = 0 if "batch" in skip else len(filters) * len(devices) * len(args.batches)
    nt = 0 if "threads" in skip else len(filters) * len(args.threads)
    nq = 0 if "quality" in skip or clean is None else (6 if gpu else 4)
    per = f"{args.trials} trials" + ("" if args.cold or not args.warmup else f" + {args.warmup} warmup")
    log(f"plan: {nb} batch-sweep runs and {nt} thread-sweep runs ({per} each), {nq} quality runs")

    batch_rows = thread_rows = []
    if "batch" not in skip:
        batch_rows = sweep(args, res, inp, "batch",
                           [(d, f, nmax, b) for f in filters for d in devices for b in args.batches])
    if "threads" not in skip:
        thread_rows = sweep(args, res, inp, "threads",
                            [("cpu", f, t, args.thread_batch) for f in filters for t in args.threads])
    write_csv(res / "runs.csv", batch_rows + thread_rows)

    qual, noise_lv, checks = [], {}, {}
    if "quality" not in skip:
        if clean is None:
            log("quality: skipped, needs noise added to the images (no clean reference)")
        else:
            qual, noise_lv, checks = quality(args, res, inp, clean, gpu)
            write_csv(res / "quality.csv", qual)

    md = summary(batch_rows, thread_rows, qual, noise_lv, checks, filters)
    (res / "summary.md").write_text(f"# imgfilter benchmark, {res.name}\n\n{md}")
    print(md)
    log(f"results in {res}")


if __name__ == "__main__":
    main()
