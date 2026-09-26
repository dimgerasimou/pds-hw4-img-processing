#!/bin/bash
#
# bench.sh - Benchmark sweeps for imgfilter.
#
# Runs imgfilter once per configuration (thread count x batch size) and
# collects one JSON file per run in a results directory, together with a
# description of the environment. Repetitions of a single configuration
# (-n trials, -w warmup trials) are done by imgfilter itself; this script
# only varies the configuration.
#
# Cache modes:
#   warm  One process per configuration with -w/-n: the warmup trial loads
#         the dataset into the page cache, the timed trials measure it hot.
#   cold  The page cache is dropped (sudo) before every run, and every
#         repetition is a separate single-trial process, so each one starts
#         cold. Warmup is disabled, since it would warm the cache.
#
# Usage: tools/bench.sh [options] <input>
#
#   -t "<list>"   Thread counts          (default: powers of 2 up to nproc, and nproc)
#   -B "<list>"   Batch sizes            (default: "256")
#   -n <trials>   Timed trials per run   (default: 5)
#   -w <wtrials>  Warmup trials per run  (default: 1, warm mode only)
#   -f <format>   Also encode and write: pgm or png (default: no output)
#   -c <mode>     Cache mode: warm or cold (default: warm)
#   -r <dir>      Results directory      (default: results/<timestamp>)
#   -h            Show this help
#
# Examples:
#   tools/bench.sh data/chest_xray_complete
#   tools/bench.sh -t "1 2 4 8 16" -B "16 64 256 1024" -f pgm data/xrays
#   tools/bench.sh -c cold -n 3 -t "8 16" data/xrays

set -euo pipefail

BIN=./bin/imgfilter

# ---------- Defaults ----------
NPROC=$(nproc)
THREADS=""
BATCHES="256"
TRIALS=5
WTRIALS=1
FORMAT=""
CACHE=warm
RESULTS=""

usage() {
	sed -n '/^# Usage:/,/^$/p' "$0" | sed 's/^# \{0,1\}//'
}

die() {
	echo "bench.sh: $*" >&2
	exit 1
}

# ---------- Arguments ----------
while getopts "t:B:n:w:f:c:r:h" opt; do
	case "$opt" in
	t) THREADS="$OPTARG" ;;
	B) BATCHES="$OPTARG" ;;
	n) TRIALS="$OPTARG" ;;
	w) WTRIALS="$OPTARG" ;;
	f) FORMAT="$OPTARG" ;;
	c) CACHE="$OPTARG" ;;
	r) RESULTS="$OPTARG" ;;
	h) usage; exit 0 ;;
	*) usage >&2; exit 1 ;;
	esac
done
shift $((OPTIND - 1))

[ $# -eq 1 ] || { usage >&2; exit 1; }
INPUT="$1"
[ -e "$INPUT" ] || die "cannot access \"$INPUT\""

case "$CACHE" in
warm|cold) ;;
*) die "cache mode must be warm or cold" ;;
esac

case "$FORMAT" in
""|pgm|png) ;;
*) die "format must be pgm or png" ;;
esac

# default thread list: 1 2 4 ... up to nproc, plus nproc itself
if [ -z "$THREADS" ]; then
	t=1
	while [ "$t" -lt "$NPROC" ]; do
		THREADS="$THREADS $t"
		t=$((t * 2))
	done
	THREADS="$THREADS $NPROC"
fi

RESULTS="${RESULTS:-results/$(date +%Y-%m-%d_%H-%M-%S)}"
mkdir -p "$RESULTS"

# Output images go next to the results, on the same disk: /tmp is often a
# tmpfs (RAM), which would both distort the write timings and eat memory.
SCRATCH="$RESULTS/scratch"

# ---------- Cold cache ----------
if [ "$CACHE" = cold ]; then
	sudo -v || die "cold cache mode needs sudo to drop the page cache"
fi

drop_caches() {
	sync
	echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
}

# ---------- Build ----------
make -s >/dev/null

# ---------- OpenMP runtime ----------
# Spread threads over physical cores first; beyond one thread per core they
# share cores (hyperthreading). Keep any values set by the caller.
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-false}"

# ---------- Environment record ----------
{
	echo "date:          $(date --iso-8601=seconds)"
	echo "host:          $(uname -n)"
	echo "kernel:        $(uname -r)"
	echo "cpu:           $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
	echo "nproc:         $NPROC"
	echo "commit:        $(git rev-parse --short HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo ' (dirty)')"
	echo "input:         $INPUT"
	echo "threads:       $THREADS"
	echo "batches:       $BATCHES"
	echo "trials:        $TRIALS"
	echo "wtrials:       $([ "$CACHE" = warm ] && echo "$WTRIALS" || echo 0)"
	echo "format:        ${FORMAT:-none}"
	echo "cache:         $CACHE"
	echo "OMP_PROC_BIND: $OMP_PROC_BIND"
	echo "OMP_PLACES:    $OMP_PLACES"
	echo "OMP_DYNAMIC:   $OMP_DYNAMIC"
	echo "governor:      $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
} > "$RESULTS/environment.txt"

# ---------- Sweep ----------
run() {
	local t="$1" b="$2" json="$3" n="$4" w="$5"
	local args=(-t "$t" -B "$b" -b "$json" -n "$n")

	[ "$w" -gt 0 ] && args+=(-w "$w")

	if [ -n "$FORMAT" ]; then
		rm -rf "$SCRATCH"
		args+=(-f "$FORMAT" -o "$SCRATCH")
	fi

	# a failing image should not abort the whole sweep; the JSON records it
	"$BIN" "${args[@]}" "$INPUT" || echo "  (imgfilter exited with $?)" >&2
}

total=0
for t in $THREADS; do for b in $BATCHES; do total=$((total + 1)); done; done
i=0

for t in $THREADS; do
	for b in $BATCHES; do
		i=$((i + 1))
		name="t${t}_B${b}"
		echo "[$i/$total] threads=$t batch=$b ($CACHE)" >&2

		if [ "$CACHE" = warm ]; then
			run "$t" "$b" "$RESULTS/$name.json" "$TRIALS" "$WTRIALS"
		else
			for k in $(seq 1 "$TRIALS"); do
				drop_caches
				run "$t" "$b" "$RESULTS/${name}_r${k}.json" 1 0
			done
		fi
	done
done

rm -rf "$SCRATCH"

# ---------- Summary ----------
if command -v jq >/dev/null; then
	echo >&2
	printf "%-22s %12s %12s %10s\n" "run" "pipeline s" "peak GB" "majflt" >&2
	for f in "$RESULTS"/*.json; do
		jq -r --arg n "$(basename "$f" .json)" \
			'[$n, .pipeline_time.median_time_s, .memory.cpu_peak_rss_gb, .memory.major_page_faults]
			 | "\(.[0]) \(.[1]) \(.[2]) \(.[3])"' "$f" |
			awk '{ printf "%-22s %12.3f %12.3f %10d\n", $1, $2, $3, $4 }' >&2
	done
fi

echo "results in $RESULTS" >&2
