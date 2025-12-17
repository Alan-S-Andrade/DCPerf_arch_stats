#!/usr/bin/env python3
"""
Run `perfspect metrics` for every detected microbenchmark unit.

This script discovers benchmarks in /proj/alanfaascache-PG0/DC2/DCPerf/benchmarks/wdl_bench,
lists their units via --bm_list, probes each unit with --benchmark --bm_regex,
and invokes `perfspect metrics` for each unit.

It writes a per-unit log file containing stdout/stderr and a small metadata header
to the output directory. The `perfspect` binary and extra args are configurable.

Usage:
  python3 run_perfspect_metrics.py --output-dir ./perfspect_out \
      --perfspect-bin perfspect --perfspect-args "--myflag value"

Note: `perfspect metrics` is invoked as:
  perfspect metrics [EXTRA ARGS] -- <benchmark-binary> --bm_regex '^<unit>$'
"""

import argparse
import shlex
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict
import re
import shutil
import json
import os


def safe_name(s: str) -> str:
    return re.sub(r'[/\\:*?"<>|]', '_', s)


class BenchmarkDiscovery:
    """Discover benchmarks and extract units using --bm_list and --bm_regex probing."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.benchmarks = {}  # name -> binary_path
        self.discover_benchmarks()

    def discover_benchmarks(self):
        """Scan for benchmark binaries in wdl_bench directory."""
        possible_paths = [
            Path("/proj/alanfaascache-PG0/DC2/DCPerf/benchmarks/wdl_bench")
        ]

        wdl_build_base = None
        for path in possible_paths:
            if path.exists():
                test_files = list(path.glob("memcpy_benchmark")) + list(path.glob("*benchmark"))
                if test_files:
                    wdl_build_base = path
                    break

        if wdl_build_base is None:
            print("No benchmark directory found. Searched:")
            for p in possible_paths:
                print(f"  - {p}")
            return

        print(f"Found benchmark base: {wdl_build_base}")
        print("Scanning for executables in:", wdl_build_base)

        for binary in sorted(wdl_build_base.glob("*")):
            if binary.is_file() and os.access(binary, os.X_OK):
                name = binary.name
                if any(p in name.lower() for p in ["benchmark", "bench", "perf"]):
                    self.benchmarks[name] = binary
                    print(f"Discovered benchmark binary: {name} -> {binary}")

    def run_benchmark_get_units(self, benchmark_binary: Path) -> Dict[str, float]:
        """Extract unit names and throughput estimates using --bm_list and --bm_regex.

        Returns mapping unit_name -> throughput (iters/sec).
        """
        benchmark_dir = benchmark_binary.parent

        # Primary path: use --bm_list to enumerate unit names
        list_cmd = [str(benchmark_binary), "--bm_list"]
        try:
            print(f"  Running to list units: {' '.join(list_cmd)}")
            res = subprocess.run(list_cmd, capture_output=True, text=True, timeout=15, cwd=benchmark_dir)
            print(f"  List returncode: {res.returncode}")
            units = {}
            if res.returncode == 0 and res.stdout and res.stdout.strip():
                unit_names = [l.strip() for l in res.stdout.splitlines() if l.strip()]
                print(f"  Found {len(unit_names)} units via --bm_list")
                for unit in unit_names:
                    # Probe each unit with --benchmark --bm_regex
                    bm_cmd = [str(benchmark_binary), "--benchmark", "--bm_regex", f"^{re.escape(unit)}$"]
                    try:
                        print(f"    Probing unit '{unit}'")
                        r2 = subprocess.run(bm_cmd, capture_output=True, text=True, timeout=20, cwd=benchmark_dir)
                        parsed = self._extract_units_from_text(r2.stdout)
                        if unit in parsed:
                            units[unit] = parsed[unit]
                        elif parsed:
                            units[unit] = list(parsed.values())[0]
                        else:
                            units[unit] = 1e6
                    except subprocess.TimeoutExpired:
                        print(f"    Probe timed out for unit '{unit}'")
                        units[unit] = 1e6
                return units
            print(f"  --bm_list produced no output; falling back to --json/text discovery")
        except subprocess.TimeoutExpired:
            print(f"  --bm_list timed out; falling back")

        # Fallback: try --json or plain run
        cmd = [str(benchmark_binary), "--json"]
        try:
            print(f"  Running to discover units: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=benchmark_dir)
            units = {}
            try:
                data = json.loads(result.stdout)
                units = self._extract_units_from_json(data)
            except json.JSONDecodeError:
                units = self._extract_units_from_text(result.stdout)

            if not units:
                out_lines = (result.stdout or "").splitlines()
                out_snip = "\n".join(out_lines[:30])
                print(f"  No units parsed. stdout (first 30 lines):\n{out_snip}")

            return units
        except subprocess.TimeoutExpired:
            print(f"  JSON discovery timed out; trying plain text fallback")
            try:
                fallback = [str(benchmark_binary)]
                result = subprocess.run(fallback, capture_output=True, text=True, timeout=15, cwd=benchmark_dir)
                units = self._extract_units_from_text(result.stdout)
                return units
            except Exception as e2:
                print(f"  Fallback failed: {e2}")
                return {}
        except Exception as e:
            print(f"  Error discovering units: {e}")
            return {}

    def _extract_units_from_json(self, data: dict) -> Dict[str, float]:
        """Extract unit names and throughput from JSON data."""
        units = {}
        if isinstance(data, dict):
            if 'benchmarks' in data and isinstance(data['benchmarks'], list):
                for bench in data['benchmarks']:
                    if 'name' in bench:
                        name = bench['name']
                        throughput = bench.get('items_per_second') or bench.get('iterations_per_second') or bench.get('real_time') or bench.get('time')
                        try:
                            units[name] = float(throughput) if throughput is not None else 1e6
                        except Exception:
                            units[name] = 1e6
            else:
                for key, value in data.items():
                    if not key.startswith('%') and isinstance(value, (int, float)):
                        units[key] = float(value)
        elif isinstance(data, list):
            for bench in data:
                if isinstance(bench, dict) and 'name' in bench:
                    name = bench['name']
                    throughput = bench.get('items_per_second') or bench.get('iterations_per_second') or bench.get('real_time') or bench.get('time')
                    try:
                        units[name] = float(throughput) if throughput is not None else 1e6
                    except Exception:
                        units[name] = 1e6
        return units

    def _extract_units_from_text(self, text: str) -> Dict[str, float]:
        """Extract unit names from text output using regex."""
        units = {}
        col_pattern = re.compile(r'^\s*(?P<name>\S+)\s+(?P<time>[\d\.]+)\s*(?P<time_unit>ns|us|ms|s)?\s+(?P<iters>[\d\.,]+)\s*(?P<mult>[KMGTP]?)\b', re.IGNORECASE)
        fallback_pattern = re.compile(r'^\s*([A-Za-z0-9_().:\-]+)\s+.*?(?:=|:)?\s*([\d.,]+)\s*([KMGT]?)(?:\s*(?:iters?/sec|items?/sec|ops?/s|ops/s|per second|it/s|it/sec))?', re.IGNORECASE)
        multipliers = {'K': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12, 'P': 1e15, '': 1}

        for line in text.splitlines():
            if not line.strip():
                continue
            m = col_pattern.match(line)
            if m:
                unit_name = m.group('name').strip()
                iters_str = m.group('iters').replace(',', '')
                mult = m.group('mult').upper() if m.group('mult') else ''
                try:
                    throughput_val = float(iters_str)
                except Exception:
                    continue
                throughput = throughput_val * multipliers.get(mult, 1)
                if unit_name and throughput > 0:
                    units[unit_name] = throughput
                continue

            m2 = fallback_pattern.match(line)
            if m2:
                unit_name = m2.group(1).strip()
                try:
                    throughput_val = float(m2.group(2).replace(',', ''))
                except Exception:
                    continue
                multiplier = m2.group(3).upper() if m2.group(3) else ''
                throughput = throughput_val * multipliers.get(multiplier, 1)
                if unit_name and throughput > 0:
                    units[unit_name] = throughput
        return units

    def calculate_measurement_duration(self, throughput: float, target_iterations: int = 50000) -> float:
        """Calculate measurement duration based on throughput."""
        if throughput <= 0:
            return 0.5
        duration = target_iterations / throughput
        return max(0.05, min(60.0, duration))


def main():
    p = argparse.ArgumentParser(description="Run perfspect metrics on discovered microbenchmark units")
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

    # Instantiate discovery (uses same logic as run_intel_pt_record.py)
    discovery = BenchmarkDiscovery(str(out_dir / 'tmp_perfspect'))

    # Filter benchmarks
    benchmarks = discovery.benchmarks
    if args.benchmark:
        benchmarks = {k: v for k, v in benchmarks.items() if args.benchmark.lower() in k.lower()}

    extra_args = shlex.split(args.perfspect_args) if args.perfspect_args else []
    if args.perfspect_output:
        extra_args.extend(["--output", args.perfspect_output])

    summary = {}
    total_start = time.time()

    for bench_name in sorted(benchmarks.keys()):
        bench_path = benchmarks[bench_name]
        print(f"\n== {bench_name} ==")

        units = discovery.run_benchmark_get_units(bench_path)
        if not units:
            print(f"  no units found for {bench_name}")
            continue

        for unit_name, throughput in sorted(units.items()):
            safe_unit = safe_name(unit_name)
            out_log = out_dir / f"{bench_name}_{safe_unit}_perfspect.log"
            out_data = out_dir / f"{bench_name}_{safe_unit}_perfspect.txt"

            # build benchmark command using --bm_regex to run the specific unit
            bench_cmd = [str(bench_path), '--bm_regex', f'^{re.escape(unit_name)}$']

            # estimate duration and timeout
            duration = discovery.calculate_measurement_duration(throughput, target_iterations=args.target_iters)
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
