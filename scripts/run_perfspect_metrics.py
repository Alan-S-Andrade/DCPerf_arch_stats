#!/usr/bin/env python3
"""
Run `perfspect metrics` for every detected folly microbenchmark unit.

This script reuses the benchmark discovery and unit-extraction logic from
`run_detailed_perf_benchmarks.py` and invokes `perfspect metrics` for each unit.

It writes a per-unit log file containing stdout/stderr and a small metadata header
to the output directory. The `perfspect` binary and extra args are configurable.

Usage:
  python3 run_perfspect_metrics.py --output-dir ./perfspect_out \
      --perfspect-bin perfspect --perfspect-args "--myflag value"

Note: `perfspect metrics` is invoked as:
  perfspect metrics [EXTRA ARGS] -- <benchmark-binary> --bm_pattern '^<unit>$'

If your `perfspect` invocation differs, pass appropriate `--perfspect-args`.
"""

import argparse
import shlex
import subprocess
import time
from pathlib import Path
from typing import Optional
import re
import shutil

# Reuse the FollyBenchmarkPerfRunner for discovery and helpers
try:
    from run_detailed_perf_benchmarks import FollyBenchmarkPerfRunner
except Exception:
    # If script is executed from another cwd, try package path
    from .run_detailed_perf_benchmarks import FollyBenchmarkPerfRunner


def safe_name(s: str) -> str:
    return re.sub(r'[/\\:*?"<>|]', '_', s)


def main():
    p = argparse.ArgumentParser(description="Run perfspect metrics on folly microbenchmark units")
    p.add_argument('--folly-test-dir', default='/users/alanuiuc/DCPerf/benchmarks/wdl_bench/wdl_sources/folly/folly/test')
    p.add_argument('--output-dir', default='./perfspect_results')
    p.add_argument('--perfspect-bin', default='perfspect')
    p.add_argument('--perfspect-args', default='', help='Extra arguments to pass to perfspect metrics (as a shell string)')
    p.add_argument('--perfspect-output', default='', help='Output path for perfspect results (passed as --output to perfspect metrics)')
    p.add_argument('--benchmark', help='Filter to a specific benchmark binary (partial match)')
    p.add_argument('--timeout-mult', type=float, default=3.0, help='Timeout multiplier for unit runs (duration * multiplier)')
    p.add_argument('--target-iters', type=int, default=50000, help='Target iterations used to estimate unit duration')

    args = p.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate runner (perf_events not used but calculate_measurement_duration is available)
    runner = FollyBenchmarkPerfRunner(args.folly_test_dir, str(out_dir / 'tmp_perfspect'), perf_events='')

    # Filter benchmarks
    benchmarks = runner.benchmarks
    if args.benchmark:
        benchmarks = {k: v for k, v in benchmarks.items() if args.benchmark.lower() in k.lower()}

    # perfspect_bin = "perfspect"
    extra_args = shlex.split(args.perfspect_args) if args.perfspect_args else []
    
    # Add --output flag if provided
    if args.perfspect_output:
        extra_args.extend(["--output", args.perfspect_output])

    # # Ensure perfspect command is available on PATH
    # if shutil.which(perfspect_bin) is None:
    #     print(f"perfspect command not found in PATH: '{perfspect_bin}'.\nPlease install perfspect or adjust PATH, or pass --perfspect-bin with the correct command.")
    #     return

    summary = {}
    total_start = time.time()

    for bench_name in sorted(benchmarks.keys()):
        bench_path = benchmarks[bench_name]
        print(f"\n== {bench_name} ==")

        units = runner.run_benchmark_get_units(bench_name)
        if not units:
            print(f"  no units found for {bench_name}")
            continue

        for unit_name, throughput in sorted(units.items()):
            safe_unit = safe_name(unit_name)
            out_log = out_dir / f"{bench_name}_{safe_unit}_perfspect.log"
            out_data = out_dir / f"{bench_name}_{safe_unit}_perfspect.txt"

            # build benchmark command
            bench_cmd = [str(bench_path), '--bm_pattern', f'^{re.escape(unit_name)}$']

            # estimate duration and timeout
            duration = runner.calculate_measurement_duration(throughput, target_iterations=args.target_iters)
            timeout = max(5.0, min(600.0, duration * args.timeout_mult))

            perfspect_cmd = ["perfspect", "metrics"] + extra_args + ["--"] + bench_cmd

            print(f"  {unit_name:<60} duration~{duration:.3f}s timeout={timeout:.1f}s -> {out_log.name}")

            start = time.time()
            try:
                proc = subprocess.run(
                    perfspect_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=bench_path.parent
                )

                elapsed = time.time() - start

                with open(out_log, 'w') as f:
                    f.write(f"Benchmark: {bench_name}\n")
                    f.write(f"Unit: {unit_name}\n")
                    f.write(f"Throughput_est: {throughput:.2e}\n")
                    f.write(f"Timeout: {timeout}\n")
                    f.write(f"Elapsed: {elapsed:.3f}\n\n")
                    f.write("=== STDOUT ===\n")
                    f.write(proc.stdout or '')
                    f.write("\n=== STDERR ===\n")
                    f.write(proc.stderr or '')

                # also save stdout-only data file for convenience
                with open(out_data, 'w') as f:
                    f.write(proc.stdout or '')

                ok = (proc.returncode == 0)
            except subprocess.TimeoutExpired:
                elapsed = time.time() - start
                with open(out_log, 'w') as f:
                    f.write(f"Timed out after {timeout}s\n")
                ok = False
            except Exception as e:
                with open(out_log, 'w') as f:
                    f.write(f"Error running perfspect: {e}\n")
                ok = False

            summary_key = f"{bench_name}::{unit_name}"
            summary[summary_key] = {'ok': ok, 'log': str(out_log), 'data': str(out_data)}

    total_elapsed = time.time() - total_start
    print("\n=== Summary ===")
    success = sum(1 for v in summary.values() if v['ok'])
    print(f"Total units processed: {len(summary)}, success: {success}")
    print(f"Output directory: {out_dir}")
    print(f"Elapsed: {total_elapsed:.1f}s")


if __name__ == '__main__':
    main()
