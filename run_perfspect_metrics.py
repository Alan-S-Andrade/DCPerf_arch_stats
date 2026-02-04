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

    def __init__(self, output_dir: str, bench_bin_dir: str = None, additional_paths: list = None,
                 binary_pattern: str = 'bench_bin_*'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bench_bin_dir = Path(bench_bin_dir) if bench_bin_dir else None
        self.additional_paths = [Path(p) for p in (additional_paths or [])]
        self.binary_pattern = binary_pattern
        self.benchmarks = {}  # name -> binary_path
        self.benchmark_type = {}  # name -> 'folly' or 'generic'
        self.discover_benchmarks()

    def discover_benchmarks(self):
        """Scan for benchmark binaries in wdl_bench directory and configured paths."""
        possible_paths = [
            Path("/myd/DCPerf/benchmarks/tiny_wdl_bench")
        ]

        wdl_build_base = None
        for path in possible_paths:
            if path.exists():
                test_files = list(path.glob("memcpy_benchmark")) + list(path.glob("*benchmark"))
                if test_files:
                    wdl_build_base = path
                    break

        if wdl_build_base is None:
            print("No folly benchmark directory found. Searched:")
            for p in possible_paths:
                print(f"  - {p}")
        else:
            print(f"Found folly benchmark base: {wdl_build_base}")
            print("Scanning for executables in:", wdl_build_base)

            for binary in sorted(wdl_build_base.glob("*")):
                if binary.is_file() and os.access(binary, os.X_OK):
                    name = binary.name
                    if any(p in name.lower() for p in ["benchmark", "bench", "perf"]):
                        self.benchmarks[name] = binary
                        self.benchmark_type[name] = 'folly'
                        print(f"Discovered folly benchmark: {name} -> {binary}")

        # Discover generic binaries from configured paths
        search_paths = [self.bench_bin_dir] + self.additional_paths
        search_paths = [p for p in search_paths if p is not None and p.exists()]
        
        if search_paths:
            print(f"\nScanning for binaries matching pattern '{self.binary_pattern}' in:")
            for search_path in search_paths:
                print(f"  - {search_path}")
            
            for search_path in search_paths:
                for binary in sorted(search_path.glob(self.binary_pattern)):
                    if binary.is_file() and os.access(binary, os.X_OK):
                        name = binary.name
                        # Skip if already discovered as folly benchmark
                        if name not in self.benchmarks:
                            self.benchmarks[name] = binary
                            self.benchmark_type[name] = 'generic'
                            print(f"Discovered generic benchmark: {name} -> {binary}")

    def run_benchmark_get_units(self, benchmark_binary: Path) -> tuple:
        """Extract unit names, throughput estimates, and probe times using --bm_list and --bm_regex.

        For folly benchmarks: returns (unit_dict, probe_times_dict) where:
          - unit_dict: mapping unit_name -> throughput (iters/sec)
          - probe_times_dict: mapping unit_name -> elapsed_time_in_seconds
        For generic binaries: returns single unit with default values
        """
        benchmark_name = benchmark_binary.name
        benchmark_type = self.benchmark_type.get(benchmark_name, 'folly')
        benchmark_dir = benchmark_binary.parent
        probe_times = {}  # Track elapsed time for each unit probe

        # For generic binaries, treat as single unit
        if benchmark_type == 'generic':
            return {benchmark_name: 1e6}, {}  # Default throughput estimate, no probe times
        
        # Try to load cached probe times first
        cached_times = self.load_probe_times(benchmark_name)
        if cached_times:
            # Skip probing and use cached times
            # Still get unit list but don't re-probe
            list_cmd = [str(benchmark_binary), "--bm_list"]
            try:
                res = subprocess.run(list_cmd, capture_output=True, text=True, timeout=15, cwd=benchmark_dir)
                if res.returncode == 0 and res.stdout and res.stdout.strip():
                    unit_names = [l.strip() for l in res.stdout.splitlines() if l.strip()]
                    units = {}
                    for unit in unit_names:
                        units[unit] = 1e6  # Default throughput estimate
                    return units, cached_times
            except subprocess.TimeoutExpired:
                pass

        # For folly benchmarks, extract units
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
                        probe_start = time.time()
                        r2 = subprocess.run(bm_cmd, capture_output=True, text=True, timeout=20, cwd=benchmark_dir)
                        probe_elapsed = time.time() - probe_start
                        
                        # Verify the unit ran successfully
                        if r2.returncode == 0:
                            # Build the perfspect command for this unit
                            perfspect_cmd_str = ' '.join(["perfspect", "metrics", "--"] + bm_cmd)
                            probe_times[unit] = (probe_elapsed, perfspect_cmd_str)
                            
                            parsed = self._extract_units_from_text(r2.stdout)
                            if unit in parsed:
                                units[unit] = parsed[unit]
                            elif parsed:
                                units[unit] = list(parsed.values())[0]
                            else:
                                units[unit] = 1e6
                        else:
                            print(f"    Probe failed for unit '{unit}' (returncode {r2.returncode})")
                    except subprocess.TimeoutExpired:
                        print(f"    Probe timed out for unit '{unit}'")
                        probe_elapsed = time.time() - probe_start
                        probe_times[unit] = (probe_elapsed, None)
                        units[unit] = 1e6
                return units, probe_times
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

            return units, {}
        except subprocess.TimeoutExpired:
            print(f"  JSON discovery timed out; trying plain text fallback")
            try:
                fallback = [str(benchmark_binary)]
                result = subprocess.run(fallback, capture_output=True, text=True, timeout=15, cwd=benchmark_dir)
                units = self._extract_units_from_text(result.stdout)
                return units, {}
            except Exception as e2:
                print(f"  Fallback failed: {e2}")
                return {}, {}
        except Exception as e:
            print(f"  Error discovering units: {e}")
            return {}, {}

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

    def load_probe_times(self, benchmark_name: str) -> Dict[str, tuple]:
        """Load cached probe times and commands from disk if they exist.
        
        Returns dict of unit_name -> (elapsed_seconds, command_list), or empty dict if not found.
        """
        if not hasattr(self, 'probe_cache_dir') or not self.probe_cache_dir:
            return {}
        
        probe_file = self.probe_cache_dir / f"{benchmark_name}_probe_times.txt"
        if not probe_file.exists():
            return {}
        
        probe_times = {}
        try:
            with open(probe_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(maxsplit=2)
                    if len(parts) >= 2:
                        unit_name = parts[0]
                        try:
                            elapsed = float(parts[1])
                            # Command is optional (parts[2] if present)
                            cmd = parts[2] if len(parts) > 2 else None
                            probe_times[unit_name] = (elapsed, cmd)
                        except ValueError:
                            continue
            if probe_times:
                print(f"  Loaded cached probe times from {probe_file.name} ({len(probe_times)} units)")
        except Exception as e:
            print(f"  Error loading cached probe times: {e}")
        
        return probe_times

    def save_probe_times(self, benchmark_name: str, probe_times: Dict[str, tuple]) -> Path:
        """Save probe times and commands to disk as {bench_name}_probe_times.txt.
        
        probe_times should be dict of unit_name -> (elapsed_seconds, perfspect_command_str)
        
        Returns path to the saved file.
        """
        if not probe_times or not hasattr(self, 'probe_cache_dir') or not self.probe_cache_dir:
            return None
        
        probe_file = self.probe_cache_dir / f"{benchmark_name}_probe_times.txt"
        with open(probe_file, 'w') as f:
            for unit_name in sorted(probe_times.keys()):
                elapsed, cmd = probe_times[unit_name]
                if cmd:
                    f.write(f"{unit_name} {elapsed:.6f} {cmd}\n")
                else:
                    f.write(f"{unit_name} {elapsed:.6f}\n")
        
        return probe_file


def main():
    p = argparse.ArgumentParser(description="Run perfspect metrics on discovered microbenchmark units")
    p.add_argument('--output-dir', default='./perfspect_results')
    p.add_argument('--bench-bin-dir', default=None, help='Directory containing binaries to run')
    p.add_argument('--additional-paths', nargs='*', default=[], 
                   help='Additional directories to scan for binaries (space-separated)')
    p.add_argument('--binary-pattern', default='bench_bin_*',
                   help='Pattern to match binary names (e.g., "bench_bin_*", "*_bench", "test_*")')
    p.add_argument('--perfspect-bin', default='perfspect')
    p.add_argument('--perfspect-args', default='', help='Extra arguments to pass to perfspect metrics (as a shell string)')
    p.add_argument('--benchmark', help='Filter to a specific benchmark binary (partial match)')
    p.add_argument('--timeout-mult', type=float, default=3.0, help='Timeout multiplier for unit runs (duration * multiplier)')
    p.add_argument('--target-iters', type=int, default=50000, help='Target iterations used to estimate unit duration')

    args = p.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate discovery with new parameters
    discovery = BenchmarkDiscovery(
        str(out_dir / 'tmp_perfspect'),
        bench_bin_dir=args.bench_bin_dir,
        additional_paths=args.additional_paths,
        binary_pattern=args.binary_pattern
    )
    
    # Set probe cache directory to output directory (persistent, not temp)
    discovery.probe_cache_dir = out_dir

    # Filter benchmarks
    benchmarks = discovery.benchmarks
    if args.benchmark:
        benchmarks = {k: v for k, v in benchmarks.items() if args.benchmark.lower() in k.lower()}

    extra_args = shlex.split(args.perfspect_args) if args.perfspect_args else []
    extra_args.extend(["--output", "perfspect_results"])

    summary = {}
    total_start = time.time()

    for bench_name in sorted(benchmarks.keys()):
        bench_path = benchmarks[bench_name]
        print(f"\n== {bench_name} ==")

        units, probe_times = discovery.run_benchmark_get_units(bench_path)
        if not units:
            print(f"  no units found for {bench_name}")
            continue
        
        # Create directory for this binary
        bench_dir = out_dir / bench_name
        bench_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if we already have all results
        probe_cache_file = out_dir / f"{bench_name}_probe_times.txt"
        if probe_cache_file.exists():
            # Count lines in probe_times.txt
            with open(probe_cache_file, 'r') as f:
                probe_lines = len([l for l in f if l.strip()])
            
            # Count existing summary files in bench_dir
            existing_summaries = list(bench_dir.glob("*_summary.csv"))
            
            if len(existing_summaries) == probe_lines and len(existing_summaries) == len(units):
                print(f"  Found {len(existing_summaries)} existing summary files matching {probe_lines} probe times and {len(units)} units - skipping perfspect")
                continue
        
        # Save probe times to disk for this benchmark
        if probe_times:
            probe_file = discovery.save_probe_times(bench_name, probe_times)
            print(f"  Saved probe times to {probe_file.name}")

        for unit_name, throughput in sorted(units.items()):
            safe_unit = safe_name(unit_name)
            # Create per-unit output directory
            unit_out_dir = bench_dir / safe_unit
            unit_out_dir.mkdir(parents=True, exist_ok=True)
            out_log = bench_dir / f"{safe_unit}_perfspect.log"
            
            # For generic binaries, run without --bm_regex; for folly use --bm_regex
            benchmark_type = discovery.benchmark_type.get(bench_name, 'folly')
            if benchmark_type == 'generic':
                bench_cmd = [str(bench_path)]
            else:
                regex_pattern = f"^{re.escape(unit_name)}$"
                bench_cmd = [str(bench_path), '--bm_regex', regex_pattern]

            # Use actual probe time if available, otherwise estimate from throughput
            if unit_name in probe_times:
                probe_data = probe_times[unit_name]
                # probe_data is tuple of (elapsed_time, perfspect_cmd_str)
                if isinstance(probe_data, tuple):
                    duration, cached_cmd = probe_data
                    duration_source = "measured"
                else:
                    # Legacy: just elapsed time
                    duration = probe_data
                    duration_source = "measured"
            else:
                duration = discovery.calculate_measurement_duration(throughput, target_iterations=args.target_iters)
                duration_source = "estimated"
            
            # Always build fresh command to ensure proper escaping (ignore cached_cmd)
            perfspect_cmd = ["perfspect", "metrics"] + extra_args + ["--"] + bench_cmd
            
            # Build perfspect command with output flag
            # Insert --output before the benchmark command (before the "--")
            dash_idx = None
            for i, arg in enumerate(perfspect_cmd):
                if arg == "--":
                    dash_idx = i
                    break
            
            if dash_idx is not None:
                # Insert --output before the "--" separator
                perfspect_cmd_with_output = perfspect_cmd[:dash_idx] + ["--output", str(unit_out_dir)] + perfspect_cmd[dash_idx:]
            else:
                # Fallback: append at end
                perfspect_cmd_with_output = perfspect_cmd + ["--output", str(unit_out_dir)]
            
            duration_display = f"{duration:.3f}s ({duration_source})"

            print(f"  {unit_name:<60} duration~{duration_display} -> {bench_name}/{safe_unit}/_summary.csv")
            print(f"  Command: {' '.join(perfspect_cmd_with_output)}")

            start = time.time()
            try:
                proc = subprocess.run(
                    perfspect_cmd_with_output,
                    capture_output=True,
                    text=True,
                    cwd=bench_path.parent
                )

                elapsed = time.time() - start
                
                # Check if metric files were successfully created
                metrics_found = "Metric files:" in proc.stdout or "Metric files:" in proc.stderr
                ok = proc.returncode == 0 and metrics_found

                with open(out_log, 'w') as f:
                    f.write(f"Benchmark: {bench_name}\n")
                    f.write(f"Type: {benchmark_type}\n")
                    f.write(f"Unit: {unit_name}\n")
                    f.write(f"Throughput_est: {throughput:.2e}\n")
                    f.write(f"Duration: {duration:.3f}s ({duration_source})\n")
                    f.write(f"Elapsed: {elapsed:.3f}\n")
                    f.write(f"Status: {'SUCCESS' if ok else 'FAILED'}\n")
                    if not metrics_found:
                        f.write(f"Warning: No 'Metric files:' found in output\n")
                    f.write("\n=== STDOUT ===\n")
                    f.write(proc.stdout or '')
                    f.write("\n=== STDERR ===\n")
                    f.write(proc.stderr or '')

                # Look for *_summary.csv files in unit_out_dir
                summary_files = list(unit_out_dir.glob("*_summary.csv"))
                if summary_files:
                    # Move the summary file to bench_dir with microbenchmark name
                    summary_csv_path = summary_files[0]
                    final_summary = bench_dir / f"{safe_unit}_summary.csv"
                    shutil.move(str(summary_csv_path), str(final_summary))
                    # Delete the per-microbenchmark directory
                    if unit_out_dir.exists():
                        shutil.rmtree(unit_out_dir)
                    ok = True
                else:
                    final_summary = None
                    ok = False

            except Exception as e:
                with open(out_log, 'w') as f:
                    f.write(f"Error running perfspect: {e}\n")
                ok = False
                final_summary = None

            summary_key = f"{bench_name}::{unit_name}"
            summary[summary_key] = {'ok': ok, 'log': str(out_log), 'summary': str(final_summary) if final_summary else None}

    total_elapsed = time.time() - total_start
    print("\n=== Summary ===")
    success = sum(1 for v in summary.values() if v['ok'])
    print(f"Total units processed: {len(summary)}, success: {success}")
    print(f"Output directory: {out_dir}")
    print(f"Elapsed: {total_elapsed:.1f}s")


if __name__ == '__main__':
    main()
