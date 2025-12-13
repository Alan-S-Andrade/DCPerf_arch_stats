#!/usr/bin/env python3
"""
Record Intel PT traces for the duration of each tiny folly microbenchmark unit.

This script discovers benchmark binaries, extracts unit names and throughput
estimates (reusing the same heuristics as the perf-stat runner), computes a
duration for each unit, and runs `sudo perf record -e intel_pt//k -a -o OUT.data sleep X`
while the benchmark unit executes. Use `--dry-run` to print commands without
executing them.
"""

import json
import subprocess
import os
import sys
import re
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import argparse
import tempfile


class FollyIntelPTRecorder:
    def __init__(self, folly_test_dir: str, output_dir: str):
        self.folly_test_dir = Path(folly_test_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.benchmarks = {}
        self.discover_benchmarks()

    def discover_benchmarks(self):
        possible_paths = [
            Path("/users/alanuiuc/DCPerf/benchmarks/wdl_bench"),
            Path.cwd() / "wdl_build",
            self.folly_test_dir.parent.parent.parent.parent / "wdl_build",
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

        for binary in sorted(wdl_build_base.glob("*")):
            if binary.is_file() and os.access(binary, os.X_OK):
                name = binary.name
                if any(p in name.lower() for p in ["benchmark", "bench", "perf"]):
                    self.benchmarks[name] = binary

    def run_benchmark_get_units(self, benchmark_binary: Path) -> Dict[str, float]:
        """Run the benchmark binary to extract unit names and throughput estimates.

        Returns mapping unit_name -> throughput (iters/sec).
        """
        benchmark_dir = benchmark_binary.parent
        cmd = [str(benchmark_binary), "--json"]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120, cwd=benchmark_dir
            )
            units = {}
            try:
                data = json.loads(result.stdout)
                units = self._extract_units_from_json(data)
            except json.JSONDecodeError:
                units = self._extract_units_from_text(result.stdout)

            return units
        except Exception:
            return {}

    def _extract_units_from_json(self, data: dict) -> Dict[str, float]:
        units = {}
        if isinstance(data, dict):
            if 'benchmarks' in data:
                for bench in data['benchmarks']:
                    if 'name' in bench:
                        name = bench['name']
                        throughput = bench.get('items_per_second', bench.get('iterations_per_second', 1e6))
                        units[name] = float(throughput) if throughput else 1e6
            else:
                for key, value in data.items():
                    if not key.startswith('%') and isinstance(value, (int, float)):
                        units[key] = float(value)
        return units

    def _extract_units_from_text(self, text: str) -> Dict[str, float]:
        units = {}
        pattern = r'([a-zA-Z0-9_\-().]+)\s+[\d.]+\s+[a-z]+\s+([\d.]+)([KMGT]?)'
        for match in re.finditer(pattern, text, re.MULTILINE):
            unit_name = match.group(1)
            throughput_val = float(match.group(2))
            multiplier = match.group(3)
            multipliers = {'K': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12}
            throughput = throughput_val * multipliers.get(multiplier, 1)
            if unit_name and throughput > 0:
                units[unit_name] = throughput
        return units

    def calculate_measurement_duration(self, throughput: float, target_iterations: int = 50000) -> float:
        if throughput <= 0:
            return 0.5
        duration = target_iterations / throughput
        duration = max(0.05, min(60.0, duration))
        return duration

    def run_unit_record(self, benchmark_binary: Path, unit_name: str, throughput: float,
                        dry_run: bool = False) -> Tuple[bool, str, float]:
        """Run perf record with intel_pt while executing the benchmark unit.

        Returns (success, perf_data_path, elapsed_time)
        """
        benchmark_name = benchmark_binary.name
        benchmark_dir = benchmark_binary.parent
        safe_unit_name = re.sub(r'[/\\:*?"<>|]', '_', unit_name)

        perf_data_file = self.output_dir / f"{benchmark_name}_{safe_unit_name}_pt.data"
        bench_out_file = self.output_dir / f"{benchmark_name}_{safe_unit_name}_bench.txt"

        duration = self.calculate_measurement_duration(throughput, target_iterations=50000)
        # Add small padding
        perf_sleep = max(1, int(duration + 1)) if duration > 1 else round(duration + 0.2, 2)
        # Cap total perf record time to at most 10 seconds per microbenchmark
        MAX_RECORD_SECS = 10.0
        if perf_sleep > MAX_RECORD_SECS:
            perf_sleep = MAX_RECORD_SECS

        perf_cmd = [
            "sudo", "perf", "record",
            "-e", "intel_pt//u",
            "-a",
            "-o", str(perf_data_file),
            "sleep", str(perf_sleep)
        ]

        benchmark_cmd = [str(benchmark_binary), "--bm_pattern", f"^{re.escape(unit_name)}$"]

        print(f"Recording: {benchmark_name} :: {unit_name} for ~{duration:.3f}s -> perf sleep {perf_sleep}")

        if dry_run:
            print("DRY-RUN: ", " ".join(perf_cmd))
            print("DRY-RUN: ", " ".join(benchmark_cmd))
            return (True, str(perf_data_file), 0.0)

        start = time.time()

        try:
            # Start perf record first
            perf_proc = subprocess.Popen(perf_cmd, cwd=benchmark_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Small delay to ensure perf record is up
            time.sleep(0.05)

            # Run benchmark; don't wait too long for it
            bench_proc = subprocess.run(
                benchmark_cmd, capture_output=True, text=True, cwd=benchmark_dir, timeout=perf_sleep + 5
            )

            # Wait for perf to finish (it will finish when sleep finishes)
            perf_stdout, perf_stderr = perf_proc.communicate(timeout=perf_sleep + 10)

            elapsed = time.time() - start

            # Save benchmark stdout and perf stderr for diagnosis
            with open(bench_out_file, 'w') as f:
                f.write(f"Benchmark: {benchmark_name}\n")
                f.write(f"Unit: {unit_name}\n")
                f.write(f"Throughput estimate: {throughput:.2e}\n")
                f.write(f"Target duration: {duration:.3f}s\n")
                f.write(f"Perf sleep: {perf_sleep}\n")
                f.write(f"Elapsed: {elapsed:.3f}s\n\n")
                f.write("Benchmark stdout:\n")
                f.write(bench_proc.stdout if bench_proc.stdout else "")
                if bench_proc.stderr:
                    f.write("\nBenchmark stderr:\n")
                    f.write(bench_proc.stderr)

            return (True, str(perf_data_file), elapsed)

        except subprocess.TimeoutExpired:
            try:
                perf_proc.kill()
            except Exception:
                pass
            return (False, "", 0.0)
        except PermissionError:
            print("Permission denied: need sudo to run perf record")
            return (False, "", 0.0)
        except Exception as e:
            print(f"Error recording unit: {e}")
            return (False, "", 0.0)

    def run_all(self, benchmark_filter: Optional[str] = None, dry_run: bool = False):
        if not self.benchmarks:
            print("No benchmarks discovered.")
            return

        selected = self.benchmarks
        if benchmark_filter:
            selected = {k: v for k, v in self.benchmarks.items() if benchmark_filter.lower() in k.lower()}

        for name in sorted(selected.keys()):
            binary = selected[name]
            units = self.run_benchmark_get_units(binary)
            if not units:
                print(f"No units for {name}")
                continue

            for unit, thr in sorted(units.items()):
                ok, perf_path, elapsed = self.run_unit_record(binary, unit, thr, dry_run=dry_run)
                if not ok:
                    print(f"Failed: {name} :: {unit}")


def main():
    parser = argparse.ArgumentParser(description="Record Intel PT traces for folly microbenchmark units")
    parser.add_argument('--folly-test-dir', default='/users/alanuiuc/DCPerf/benchmarks/wdl_bench/wdl_sources/folly/folly/test')
    parser.add_argument('--output-dir', default='./pt_records')
    parser.add_argument('--benchmark', help='Filter to a specific benchmark (partial match)')
    parser.add_argument('--dry-run', action='store_true', help='Print commands instead of running')

    args = parser.parse_args()

    outdir = Path(args.output_dir)
    if not outdir.is_absolute():
        outdir = Path.cwd() / outdir

    recorder = FollyIntelPTRecorder(args.folly_test_dir, str(outdir))
    recorder.run_all(args.benchmark, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
