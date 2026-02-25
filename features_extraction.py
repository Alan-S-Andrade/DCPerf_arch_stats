#!/usr/bin/env python3
"""
Feature extraction pipeline for a single benchmark command.

Runs all three metric collectors (perf stat, perfspect TMA, Intel PT trace
analysis) on the supplied command and writes a unified JSON file titled
  <benchmark_name>_feature_extraction.json

Usage:
  python3 features_extraction.py "sysbench cpu --cpu-max-prime=20000 --threads=8 --time=8 run"

  # Or with options:
  python3 features_extraction.py --name my_bench --output-dir ./out \\
      "sysbench cpu --cpu-max-prime=20000 --threads=8 --time=8 run"
"""

import argparse
import json
import shlex
import sys
import time
from collections import OrderedDict
from pathlib import Path


def _benchmark_name(cmd_list):
    """Derive a short name from the first token of the command."""
    return Path(cmd_list[0]).stem if cmd_list else "unknown"


def main():
    ap = argparse.ArgumentParser(
        description="Collect perf-stat, perfspect-TMA, and Intel-PT trace "
                    "features for a single benchmark command.",
    )
    ap.add_argument(
        "single_run_cmd",
        help='Benchmark command as a single quoted string, e.g. '
             '"sysbench cpu --cpu-max-prime=20000 --threads=8 --time=8 run"',
    )
    ap.add_argument("--name", default=None,
                    help="Override auto-detected benchmark name")
    ap.add_argument("--output-dir", default="./feature_extraction",
                    help="Base directory for intermediate files and final JSON")
    ap.add_argument("--perfspect-bin", default="perfspect",
                    help="Path to the perfspect binary")
    ap.add_argument("--events", default=None,
                    help="Override perf stat PMU events (comma-separated)")
    ap.add_argument("--skip-perf-stat", action="store_true")
    ap.add_argument("--skip-perfspect", action="store_true")
    ap.add_argument("--skip-intel-pt", action="store_true")
    args = ap.parse_args()

    # ---- parse the command string into a list ----
    cmd_list = shlex.split(args.single_run_cmd)
    if not cmd_list:
        print("Error: empty command")
        sys.exit(1)

    bench_name = args.name or _benchmark_name(cmd_list)

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Put the repo on sys.path so we can import the helper modules
    repo_dir = Path(__file__).resolve().parent
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    all_metrics: OrderedDict = OrderedDict()
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. perf stat  →  Elapsed time, IPC, cycles, instructions, caches …
    # ------------------------------------------------------------------
    if not args.skip_perf_stat:
        print(f"\n{'='*70}")
        print("STEP 1/3  perf stat (hardware counters)")
        print(f"{'='*70}")
        try:
            from run_detailed_perf_benchmarks import run_single_cmd_perf_stat

            perf_dir = str(out_dir / "perf_stat")
            m = run_single_cmd_perf_stat(cmd_list, args.events, perf_dir)
            all_metrics.update(m)
            print(f"  → collected {len(m)} counters")
        except Exception as exc:
            print(f"  ✗ perf stat failed: {exc}")
    else:
        print("\n[skip] perf stat")

    # ------------------------------------------------------------------
    # 2. perfspect  →  TMA_* top-down metrics
    # ------------------------------------------------------------------
    if not args.skip_perfspect:
        print(f"\n{'='*70}")
        print("STEP 2/3  perfspect (TMA top-down analysis)")
        print(f"{'='*70}")
        try:
            from run_perfspect_metrics import run_single_cmd_perfspect

            ps_dir = str(out_dir / "perfspect")
            m = run_single_cmd_perfspect(cmd_list, args.perfspect_bin, [], ps_dir)
            all_metrics.update(m)
            print(f"  → collected {len(m)} TMA metrics")
        except Exception as exc:
            print(f"  ✗ perfspect failed: {exc}")
    else:
        print("\n[skip] perfspect")

    # ------------------------------------------------------------------
    # 3. Intel PT  →  block_size_P*, jumps_P*, family::*_P*, …
    # ------------------------------------------------------------------
    if not args.skip_intel_pt:
        print(f"\n{'='*70}")
        print("STEP 3/3  Intel PT (instruction trace analysis)")
        print(f"{'='*70}")
        try:
            from run_intel_pt_record import run_single_cmd_intel_pt

            pt_dir = str(out_dir / "intel_pt")
            m = run_single_cmd_intel_pt(cmd_list, pt_dir)
            all_metrics.update(m)
            print(f"  → collected {len(m)} trace percentile metrics")
        except Exception as exc:
            print(f"  ✗ Intel PT failed: {exc}")
    else:
        print("\n[skip] Intel PT")

    # ------------------------------------------------------------------
    # Write the final JSON
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    json_path = out_dir / f"{bench_name}_feature_extraction.json"
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

    print(f"\n{'='*70}")
    print(f"DONE  {len(all_metrics)} total metrics  ({elapsed:.1f}s)")
    print(f"Output → {json_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
