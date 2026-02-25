#!/usr/bin/env python3
"""
Feature Extraction Orchestrator

Runs all three metric collection tools (perf stat, Intel PT, perfspect) on a
single command-line benchmark and produces a unified JSON feature-extraction file.

The output JSON contains:
  - Hardware counter metrics from perf stat (cycles, instructions, IPC, cache misses, etc.)
  - TMA (Top-down Microarchitecture Analysis) metrics from perfspect
  - Instruction trace percentile metrics from Intel PT (block sizes, jump distances,
    branch run lengths, RAW dependency distances, instruction family distributions)

Usage:
  python3 run_feature_extraction.py --cmd sysbench cpu --cpu-max-prime=20000 --threads=8 --time=8 run

  Output: sysbench_feature_extraction.json

Options:
  --output-dir DIR    Base directory for intermediate and final output (default: ./feature_extraction)
  --benchmark-name N  Override the auto-detected benchmark name
  --perfspect-bin P   Path to perfspect binary (default: perfspect)
  --events E          PMU events for perf stat (comma-separated)
  --skip-perf-stat    Skip the perf stat collection step
  --skip-intel-pt     Skip the Intel PT collection step
  --skip-perfspect    Skip the perfspect metrics collection step
"""

import argparse
import json
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
            perf_metrics = run_single_cmd_perf_stat(cmd_list, args.events, perf_stat_dir)
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
            tma_metrics = run_single_cmd_perfspect(
                cmd_list, args.perfspect_bin, [], perfspect_dir
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
            pt_metrics = run_single_cmd_intel_pt(cmd_list, pt_dir)
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
