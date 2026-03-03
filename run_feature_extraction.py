#!/usr/bin/env python3
"""
Feature Extraction Orchestrator

Runs all three metric collection tools (perf stat, Intel PT, perfspect) on a
single command-line benchmark and produces a unified JSON feature-extraction file.

Cloud workload timing support:
  - Optional warmup delay (default 0s): start feature extraction after warmup
  - Optional extraction window (default 0s): extract for N seconds total
  - Convenience flag --cloud-mode sets warmup=30, window=10

IMPORTANT:
  This orchestrator propagates the timing window to the tool runners via
  environment variables and optional kwargs (when supported):
    FEATURE_WARMUP_S, FEATURE_WINDOW_S

  Your underlying collectors (run_detailed_perf_benchmarks.py,
  run_perfspect_metrics.py, run_intel_pt_record.py) must respect these variables
  (or accept the kwargs) for the windowing to actually take effect.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import OrderedDict


def extract_benchmark_name(cmd_list):
    """Extract a short benchmark name from the command list.

    Uses the basename of the first token (the executable).
    Examples:
      ['sysbench', 'cpu', ...] -> 'sysbench'
      ['/usr/bin/my_bench', ...] -> 'my_bench'
    """
    if not cmd_list:
        return "unknown"
    exe = Path(cmd_list[0]).name
    # Remove common extensions
    for ext in ['.py', '.sh', '.bin', '.exe']:
        if exe.endswith(ext):
            exe = exe[:-len(ext)]
    return exe


def _set_feature_window_env(warmup_s: float, window_s: float):
    """
    Propagate window settings to downstream scripts. This is the lowest-friction way
    to thread the policy through without breaking existing function signatures.
    """
    os.environ["FEATURE_WARMUP_S"] = str(float(warmup_s))
    os.environ["FEATURE_WINDOW_S"] = str(float(window_s))


def _maybe_call_with_window(fn, *args, warmup_s: float, window_s: float, **kwargs):
    """
    Call downstream runner functions in a backward-compatible way:
      - Prefer passing warmup/window kwargs if the callee accepts them.
      - Otherwise fall back to calling without them (env vars still set).
    """
    try:
        return fn(*args, feature_warmup_s=warmup_s, feature_window_s=window_s, **kwargs)
    except TypeError:
        # Downstream runner doesn't accept these kwargs; rely on env vars or old behavior.
        return fn(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Run all feature extraction tools on a single benchmark command",
        usage="%(prog)s [options] --cmd <benchmark command...>"
    )
    parser.add_argument(
        '--output-dir',
        default='./feature_extraction',
        help='Base directory for intermediate files and final JSON output (default: ./feature_extraction)'
    )
    parser.add_argument(
        '--benchmark-name',
        default=None,
        help='Override the auto-detected benchmark name used in the output filename'
    )
    parser.add_argument(
        '--perfspect-bin',
        default='perfspect',
        help='Path to the perfspect binary (default: perfspect)'
    )
    parser.add_argument(
        '--events',
        default='cycles,instructions,L1-icache-load-misses,iTLB-load-misses,'
                'L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,'
                'LLC-load-misses,branch-load-misses,branch-misses,r2424',
        help='Comma-separated PMU events for perf stat'
    )
    parser.add_argument('--skip-perf-stat', action='store_true', help='Skip perf stat collection')
    parser.add_argument('--skip-intel-pt', action='store_true', help='Skip Intel PT collection')
    parser.add_argument('--skip-perfspect', action='store_true', help='Skip perfspect metrics collection')

    # ---------------- NEW: cloud windowing knobs ----------------
    parser.add_argument(
        '--feature-warmup-seconds',
        type=float,
        default=0.0,
        help='Delay (seconds) after benchmark start before feature extraction begins (default: 0)'
    )
    parser.add_argument(
        '--feature-window-seconds',
        type=float,
        default=0.0,
        help='Total duration (seconds) to run feature extraction once started. 0 means full run / existing behavior (default: 0)'
    )
    parser.add_argument(
        '--cloud-mode',
        action='store_true',
        help='Convenience mode for cloud workloads: warmup=30s, window=10s'
    )
    # ------------------------------------------------------------

    parser.add_argument(
        '--cmd',
        nargs=argparse.REMAINDER,
        default=None,
        help='The benchmark command to run. Everything after --cmd is the command. '
             'Example: --cmd sysbench cpu --cpu-max-prime=20000 --threads=8 --time=8 run'
    )

    args = parser.parse_args()

    # Parse command
    cmd_list = args.cmd
    if not cmd_list:
        parser.print_help()
        print("\nError: --cmd is required. Provide the benchmark command to run.")
        sys.exit(1)
    # Strip leading '--' if present (argparse REMAINDER quirk)
    if cmd_list and cmd_list[0] == '--':
        cmd_list = cmd_list[1:]
    if not cmd_list:
        print("Error: --cmd requires a command to run")
        sys.exit(1)

    # ---------------- NEW: resolve window parameters ----------------
    warmup_s = float(args.feature_warmup_seconds)
    window_s = float(args.feature_window_seconds)
    if args.cloud_mode:
        warmup_s = 30.0
        window_s = 10.0

    if warmup_s < 0 or window_s < 0:
        print("Error: --feature-warmup-seconds and --feature-window-seconds must be >= 0")
        sys.exit(1)

    # Publish to env for downstream tools (even if they don't take kwargs)
    _set_feature_window_env(warmup_s, window_s)
    # ---------------------------------------------------------------

    # Determine benchmark name and output paths
    benchmark_name = args.benchmark_name or extract_benchmark_name(cmd_list)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print(f"Feature Extraction Pipeline")
    print(f"{'='*70}")
    print(f"Benchmark name: {benchmark_name}")
    print(f"Command: {' '.join(cmd_list)}")
    print(f"Output directory: {output_dir}")
    # ---------------- NEW: print window policy ----------------
    if window_s > 0 or warmup_s > 0:
        print(f"Feature extraction timing: warmup={warmup_s:.1f}s, window={window_s:.1f}s")
        print(f"  (exported: FEATURE_WARMUP_S={os.environ.get('FEATURE_WARMUP_S')}, "
              f"FEATURE_WINDOW_S={os.environ.get('FEATURE_WINDOW_S')})")
    else:
        print("Feature extraction timing: (default) full run / existing behavior")
    # ----------------------------------------------------------
    print(f"{'='*70}\n")

    # Import the three collection modules
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))

    all_metrics = OrderedDict()
    total_start = time.time()

    # ---- Step 1: perf stat (hardware counters) ----
    if not args.skip_perf_stat:
        print("\n" + "="*70)
        print("STEP 1/3: Collecting perf stat hardware counters")
        print("="*70)
        try:
            from run_detailed_perf_benchmarks import run_single_cmd_perf_stat
            perf_stat_dir = str(output_dir / "perf_stat")

            # NEW: pass window knobs if supported; otherwise env vars apply (or old behavior)
            perf_metrics = _maybe_call_with_window(
                run_single_cmd_perf_stat,
                cmd_list,
                args.events,
                perf_stat_dir,
                warmup_s=warmup_s,
                window_s=window_s,
            )

            all_metrics.update(perf_metrics)
            print(f"\n[Step 1] Collected {len(perf_metrics)} perf stat metrics")
        except Exception as e:
            print(f"\n[Step 1] ERROR: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[Step 1] Skipped perf stat collection")

    # ---- Step 2: perfspect (TMA metrics) ----
    if not args.skip_perfspect:
        print("\n" + "="*70)
        print("STEP 2/3: Collecting perfspect TMA metrics")
        print("="*70)
        try:
            from run_perfspect_metrics import run_single_cmd_perfspect
            perfspect_dir = str(output_dir / "perfspect")

            # NEW: pass window knobs if supported; otherwise env vars apply (or old behavior)
            tma_metrics = _maybe_call_with_window(
                run_single_cmd_perfspect,
                cmd_list,
                args.perfspect_bin,
                [],
                perfspect_dir,
                warmup_s=warmup_s,
                window_s=window_s,
            )

            all_metrics.update(tma_metrics)
            print(f"\n[Step 2] Collected {len(tma_metrics)} TMA metrics")
        except Exception as e:
            print(f"\n[Step 2] ERROR: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[Step 2] Skipped perfspect collection")

    # ---- Step 3: Intel PT (instruction trace analysis) ----
    if not args.skip_intel_pt:
        print("\n" + "="*70)
        print("STEP 3/3: Collecting Intel PT instruction trace metrics")
        print("="*70)
        try:
            from run_intel_pt_record import run_single_cmd_intel_pt
            pt_dir = str(output_dir / "intel_pt")

            # NEW: pass window knobs if supported; otherwise env vars apply (or old behavior)
            pt_metrics = _maybe_call_with_window(
                run_single_cmd_intel_pt,
                cmd_list,
                pt_dir,
                warmup_s=warmup_s,
                window_s=window_s,
            )

            all_metrics.update(pt_metrics)
            print(f"\n[Step 3] Collected {len(pt_metrics)} trace metrics")
        except Exception as e:
            print(f"\n[Step 3] ERROR: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[Step 3] Skipped Intel PT collection")

    # ---- Write final JSON ----
    total_elapsed = time.time() - total_start

    output_json = output_dir / f"{benchmark_name}_feature_extraction.json"
    with open(output_json, 'w') as f:
        json.dump(all_metrics, f, indent=4)

    print(f"\n{'='*70}")
    print(f"FEATURE EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"Total metrics collected: {len(all_metrics)}")
    print(f"Total elapsed time: {total_elapsed:.1f}s")
    print(f"Output file: {output_json}")
    print(f"{'='*70}\n")

    # Print a preview of the metrics
    print("Preview of collected metrics:")
    for i, (k, v) in enumerate(all_metrics.items()):
        if i >= 10:
            print(f"  ... and {len(all_metrics) - 10} more")
            break
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()