#!/usr/bin/env python3
"""
Benchmark script for hw1 convolution.

Usage:
    python3 benchmark/benchmark.py --small testfiles/lena.bmp --large testfiles/mars_large.bmp

Requirements:
    pip install numpy scipy matplotlib
"""

import argparse
import csv
import math
import os
import subprocess
import sys
import time
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hw1_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

PARALLEL_TYPES = ["horizontal", "vertical", "block", "pixel"]
# block requires perfect-square thread counts (sqrt must be integer)
BLOCK_THREAD_COUNTS = [1, 4, 9, 16]
MIN_RECOMMENDED_RUNS = 21
REL_STD_WARN_PCT = 10.0
REL_STD_OK_PCT = 5.0


def check_environment(binary):
    print("=== Environment check ===")

    print(f"  CPUs available: {os.cpu_count()}")
    print(f"  Timer resolution: {time.get_clock_info('monotonic').resolution * 1e9:.0f} ns")

    gov_path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    if os.path.exists(gov_path):
        with open(gov_path) as f:
            gov = f.read().strip()
        if gov != "performance":
            print(f"  WARNING: CPU governor is '{gov}' — results may be noisy")
            print("           Fix: sudo cpupower frequency-set -g performance")
        else:
            print(f"  CPU governor: {gov} OK")
    else:
        print("  CPU governor: not available (VM / no cpufreq driver)")

    build_dir = os.path.dirname(os.path.abspath(binary))
    cmake_cache = os.path.join(build_dir, "CMakeCache.txt")
    if os.path.exists(cmake_cache):
        found_build_type = False
        with open(cmake_cache) as f:
            for line in f:
                if line.startswith("CMAKE_BUILD_TYPE:"):
                    found_build_type = True
                    build_type = line.split("=", 1)[1].strip()
                    if build_type.upper() != "RELEASE":
                        shown = build_type or "<empty>"
                        print(f"  WARNING: build type is '{shown}' — benchmark needs Release!")
                        print("           Fix: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build")
                    else:
                        print(f"  Build type: {build_type} OK")
                    break
        if not found_build_type:
            print("  WARNING: CMAKE_BUILD_TYPE not found in cache — check that this is an optimized build")
    print()


def run_once(binary, input_img, output_img, filter_name, parallel_type=None, threads=None, repeat=1):
    cmd = [binary, input_img, output_img, filter_name]
    if parallel_type:
        cmd.append(f"--parallel={parallel_type}")
    if threads is not None:
        cmd.append(f"--threads={threads}")
    if repeat > 1:
        cmd.append(f"--repeat={repeat}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {' '.join(cmd)}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    for line in result.stdout.splitlines():
        if line.startswith("TIME_MS:"):
            return float(line.split(":", 1)[1].strip())
    raise RuntimeError(f"No TIME_MS in output:\n{result.stdout}")


def collect(binary, input_img, output_img, filter_name, n_runs,
            parallel_type=None, threads=None, repeat=1, warmup=3):
    for _ in range(warmup):
        run_once(binary, input_img, output_img, filter_name, parallel_type, threads, repeat)
    return np.array([
        run_once(binary, input_img, output_img, filter_name, parallel_type, threads, repeat)
        for _ in range(n_runs)
    ])


def _normality_tests(times):
    tests = {}
    if len(times) >= 8:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, tests["normaltest"] = stats.normaltest(times)
    else:
        tests["normaltest"] = None

    if len(times) >= 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, tests["shapiro"] = stats.shapiro(times)
    else:
        tests["shapiro"] = None
    return tests


def analyze(times, label):
    # Drop single outlier (Grubbs-style: only if exactly one extreme point)
    dropped_outliers = 0
    if len(times) >= 3 and np.std(times, ddof=1) > 0:
        z = np.abs(stats.zscore(times))
        outlier_idx = np.where(z > 3)[0]
        if len(outlier_idx) == 1:
            print(f"  [{label}] dropping 1 outlier ({times[outlier_idx[0]]:.1f} ms)")
            times = np.delete(times, outlier_idx)
            dropped_outliers = 1
        elif len(outlier_idx) > 1:
            print(f"  [{label}] WARNING: {len(outlier_idx)} outliers — check environment!")

    mean = np.mean(times)
    std = np.std(times, ddof=1)
    rel_std = std / mean * 100
    rel_sem = stats.sem(times) / mean * 100

    normality = _normality_tests(times)
    available_p = [p for p in normality.values() if p is not None]
    if available_p and all(p <= 0.05 for p in available_p):
        pretty = ", ".join(f"{name} p={p:.3f}" for name, p in normality.items() if p is not None)
        print(f"  [{label}] WARNING: non-normal data ({pretty})")
    if rel_std > REL_STD_WARN_PCT:
        print(f"  [{label}] WARNING: high relative std {rel_std:.1f}%")
    elif rel_std > REL_STD_OK_PCT:
        print(f"  [{label}] note: relative std {rel_std:.1f}%")

    ci = stats.t.ppf(0.975, df=len(times) - 1) * stats.sem(times)
    return {
        "mean": mean,
        "std": std,
        "rel_std_pct": rel_std,
        "sem": stats.sem(times),
        "rel_sem_pct": rel_sem,
        "ci": ci,
        "n": len(times),
        "dropped_outliers": dropped_outliers,
        "normaltest_p": normality["normaltest"],
        "shapiro_p": normality["shapiro"],
        "times": times,
    }


def _round_result(mean, ci):
    if ci == 0:
        return mean, ci
    mag = math.floor(math.log10(ci))
    first_digit = int(ci / 10 ** mag)
    precision = -mag + 1 if first_digit == 1 else -mag
    return round(mean, precision), round(ci, precision)


def fmt(mean, ci):
    m, c = _round_result(mean, ci)
    if c == int(c) and m == int(m):
        return f"{int(m)} ± {int(c)}"
    return f"{m} ± {c}"


def save_histograms(hist_dict, title, path):
    n = len(hist_dict)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, (label, times) in zip(axes, hist_dict.items()):
        ax.hist(times, bins=10, edgecolor="black")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("ms")
        ax.set_ylabel("count")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  saved {path}")


def save_speedup_plot(all_results, thread_counts, out_dir):
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (img_label, results) in zip(axes, all_results.items()):
        mean_seq = results["sequential"]["mean"]
        max_threads = 0

        for ptype in PARALLEL_TYPES:
            tc = BLOCK_THREAD_COUNTS if ptype == "block" else thread_counts
            points = [(t, mean_seq / results[f"{ptype}/{t}t"]["mean"])
                      for t in tc if f"{ptype}/{t}t" in results]
            if not points:
                continue
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker="o", label=ptype)
            max_threads = max(max_threads, max(xs))

        if max_threads > 1:
            ax.plot([1, max_threads], [1, max_threads], "--", color="gray",
                    linewidth=1, label="ideal")

        ax.set_title(img_label)
        ax.set_xlabel("threads")
        ax.set_ylabel("speedup")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Speedup vs threads", fontsize=12)
    plt.tight_layout()
    path = f"{out_dir}/speedup.png"
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"Saved: {path}")


def run_image(binary, img_path, img_label, filter_name, thread_counts, n_runs, out_dir,
             repeat=1, repeat_parallel=1):
    output_tmp = "/tmp/_benchmark_out.bmp"
    results = {}   
    hist_data = {}

    # --- sequential ---
    label = "sequential"
    print(f"  {label} ...")
    raw = collect(binary, img_path, output_tmp, filter_name, n_runs, repeat=repeat)
    stats_seq = analyze(raw, label)
    results[label] = stats_seq
    hist_data[label] = stats_seq["times"]
    print(f"    {fmt(stats_seq['mean'], stats_seq['ci'])} ms; std={stats_seq['rel_std_pct']:.1f}%")

    # --- parallel ---
    for ptype in PARALLEL_TYPES:
        tc = BLOCK_THREAD_COUNTS if ptype == "block" else thread_counts
        hist_group = {}
        for n in tc:
            label = f"{ptype}/{n}t"
            print(f"  {label} ...")
            raw = collect(binary, img_path, output_tmp, filter_name, n_runs,
                          parallel_type=ptype, threads=n, repeat=repeat_parallel)
            run_stats = analyze(raw, label)
            results[label] = run_stats
            hist_group[f"{n}t"] = run_stats["times"]
            speedup = stats_seq["mean"] / run_stats["mean"]
            print(f"    {fmt(run_stats['mean'], run_stats['ci'])} ms; std={run_stats['rel_std_pct']:.1f}%  speedup {speedup:.2f}x")

        save_histograms(hist_group,
                        f"{ptype} — {img_label}",
                        f"{out_dir}/hist_{img_label}_{ptype}.png")

    save_histograms({"sequential": hist_data["sequential"]},
                    f"sequential — {img_label}",
                    f"{out_dir}/hist_{img_label}_sequential.png")

    return results


def print_table(results, img_label):
    mean_seq = results["sequential"]["mean"]
    print(f"\n{'':=<60}")
    print(f"  Results: {img_label}")
    print(f"{'':=<60}")
    print(f"  {'Config':<28} {'Time (ms)':<22} {'Std':<8} {'Speedup'}")
    print(f"  {'-'*28} {'-'*22} {'-'*8} {'-'*8}")
    for label, run_stats in results.items():
        speedup = mean_seq / run_stats["mean"]
        rel_std = f"{run_stats['rel_std_pct']:.1f}%"
        print(f"  {label:<28} {fmt(run_stats['mean'], run_stats['ci']):<22} {rel_std:<8} {speedup:.2f}x")


def save_csv(all_results, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "image", "config", "n", "mean_ms", "std_ms", "rel_std_pct",
            "ci95_ms", "rel_ci95_pct", "normaltest_p", "shapiro_p",
            "dropped_outliers",
        ])
        for img_label, results in all_results.items():
            for config, run_stats in results.items():
                rel_ci = run_stats["ci"] / run_stats["mean"] * 100
                w.writerow([
                    img_label,
                    config,
                    run_stats["n"],
                    f"{run_stats['mean']:.6f}",
                    f"{run_stats['std']:.6f}",
                    f"{run_stats['rel_std_pct']:.3f}",
                    f"{run_stats['ci']:.6f}",
                    f"{rel_ci:.3f}",
                    "" if run_stats["normaltest_p"] is None else f"{run_stats['normaltest_p']:.6g}",
                    "" if run_stats["shapiro_p"] is None else f"{run_stats['shapiro_p']:.6g}",
                    run_stats["dropped_outliers"],
                ])
    print(f"\nSaved: {path}")


def validate_args(args):
    if args.runs < MIN_RECOMMENDED_RUNS:
        print(f"WARNING: --runs={args.runs}; cheat sheet recommends more than 20 measurements, e.g. 40")
    if args.runs < 3:
        print("ERROR: --runs must be at least 3 for statistics", file=sys.stderr)
        sys.exit(1)
    for path_arg, path in (("--small", args.small), ("--large", args.large)):
        if not os.path.isfile(path):
            print(f"ERROR: {path_arg} image not found: {path}", file=sys.stderr)
            sys.exit(1)
    for name, value in (
        ("--repeat-small", args.repeat_small),
        ("--repeat-parallel-small", args.repeat_parallel_small),
        ("--repeat-large", args.repeat_large),
        ("--repeat-parallel-large", args.repeat_parallel_large),
    ):
        if value < 1:
            print(f"ERROR: {name} must be >= 1", file=sys.stderr)
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Benchmark hw1 convolution")
    parser.add_argument("--binary", default="./build/hw1")
    parser.add_argument("--small", required=True, help="path to small BMP")
    parser.add_argument("--large", required=True, help="path to large BMP")
    parser.add_argument("--filter", default="gaussianBlur")
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--threads", default="1,2,4,8,16",
                        help="comma-separated thread counts (not used for block)")
    parser.add_argument("--repeat-small", type=int, default=50,
                        help="repeat sequential N times per run for small image (default: 50)")
    parser.add_argument("--repeat-parallel-small", type=int, default=10,
                        help="repeat parallel N times per run for small image (default: 10)")
    parser.add_argument("--repeat-large", type=int, default=1,
                        help="repeat filter N times per run for large image (default: 1)")
    parser.add_argument("--repeat-parallel-large", type=int, default=1,
                        help="repeat parallel filter N times per run for large image (default: 1)")
    parser.add_argument("--out", default="benchmark_results")
    args = parser.parse_args()

    if not os.path.isfile(args.binary):
        print(f"ERROR: binary not found: {args.binary}", file=sys.stderr)
        print("Build first:  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build", file=sys.stderr)
        sys.exit(1)

    validate_args(args)
    check_environment(args.binary)

    try:
        thread_counts = [int(t) for t in args.threads.split(",")]
    except ValueError:
        print(f"ERROR: invalid --threads value: {args.threads}", file=sys.stderr)
        sys.exit(1)
    if any(t < 1 for t in thread_counts):
        print("ERROR: all thread counts must be >= 1", file=sys.stderr)
        sys.exit(1)
    if os.cpu_count() is not None and max(thread_counts + BLOCK_THREAD_COUNTS) > os.cpu_count():
        print(f"WARNING: measuring with more threads than CPUs ({os.cpu_count()}) can distort results")
    os.makedirs(args.out, exist_ok=True)

    repeat_seq   = {"small": args.repeat_small,          "large": args.repeat_large}
    repeat_par   = {"small": args.repeat_parallel_small,  "large": args.repeat_parallel_large}
    images = {"small": args.small, "large": args.large}
    all_results = {}

    for img_label, img_path in images.items():
        print(f"\n=== {img_label}: {img_path} "
              f"(seq repeat={repeat_seq[img_label]}, par repeat={repeat_par[img_label]}) ===")
        all_results[img_label] = run_image(
            args.binary, img_path, img_label,
            args.filter, thread_counts, args.runs, args.out,
            repeat=repeat_seq[img_label], repeat_parallel=repeat_par[img_label]
        )
        print_table(all_results[img_label], img_label)

    save_csv(all_results, f"{args.out}/results.csv")
    save_speedup_plot(all_results, thread_counts, args.out)


if __name__ == "__main__":
    main()
