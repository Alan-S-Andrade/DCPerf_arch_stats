#!/usr/bin/env python3

import json
import subprocess
import os
import sys
import re
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Set
import argparse
import tempfile
import pickle


class FollyIntelPTRecorder:
    def __init__(self, folly_test_dir: str, output_dir: str):
        self.folly_test_dir = Path(folly_test_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.benchmarks = {}
        self.checkpoint_file = self.output_dir / ".resume_checkpoint.pkl"
        # Discover benchmarks FIRST so we have benchmark names for parsing filenames
        self.discover_benchmarks()
        # Then load checkpoint, which will use discovered benchmarks to parse filenames
        self.completed_work = self._load_checkpoint()

    def _load_checkpoint(self) -> Set[str]:
        """Load the set of completed (benchmark, unit) pairs from checkpoint and existing .data files."""
        completed = set()
        
        # First, try to load from checkpoint file if it exists
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    completed = pickle.load(f)
                    print(f"Loaded checkpoint: {len(completed)} completed work items")
            except Exception as e:
                print(f"Warning: Failed to load checkpoint: {e}. Scanning directory...")
        
        # Also scan for existing .data files in output_dir
        existing_files = list(self.output_dir.glob("*_pt.data"))
        if existing_files:
            print(f"Found {len(existing_files)} existing .data files in {self.output_dir}")
            
            # Parse filenames to extract benchmark and unit names
            # Format: {benchmark_name}_{safe_unit_name}_pt.data
            for data_file in existing_files:
                filename = data_file.name
                # Remove the _pt.data suffix
                if filename.endswith("_pt.data"):
                    base_name = filename[:-8]  # Remove "_pt.data"
                    
                    # Try to match against discovered benchmarks to split benchmark_name from unit_name
                    matched = False
                    for bench_name in sorted(self.benchmarks.keys(), key=len, reverse=True):
                        if base_name.startswith(bench_name + "_"):
                            unit_name = base_name[len(bench_name) + 1:]
                            work_key = f"{bench_name}::{unit_name}"
                            completed.add(work_key)
                            matched = True
                            print(f"  Found completed: {bench_name} :: {unit_name}")
                            break
                    
                    if not matched:
                        print(f"  Warning: Could not parse filename: {filename}")
        
        if completed:
            print(f"Total completed work items: {len(completed)}")
        return completed

    def _save_checkpoint(self):
        """Save the set of completed work to checkpoint file."""
        try:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(self.completed_work, f)
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")

    def _mark_completed(self, benchmark_name: str, unit_name: str):
        """Mark a benchmark/unit pair as completed and save checkpoint."""
        work_key = f"{benchmark_name}::{unit_name}"
        self.completed_work.add(work_key)
        self._save_checkpoint()

    def _is_completed(self, benchmark_name: str, unit_name: str) -> bool:
        """Check if a benchmark/unit pair has already been processed."""
        work_key = f"{benchmark_name}::{unit_name}"
        return work_key in self.completed_work

    def discover_benchmarks(self):
        possible_paths = [
            Path("/mydata/DCPerf/benchmarks/wdl_bench")
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
        """Run the benchmark binary to extract unit names and throughput estimates.

        Returns mapping unit_name -> throughput (iters/sec).
        """
        benchmark_dir = benchmark_binary.parent

        # Primary path: many benchmarks support --bm_list to enumerate unit names
        list_cmd = [str(benchmark_binary), "--bm_list"]
        try:
            print(f"Running to list units: {' '.join(list_cmd)} (cwd={benchmark_dir})")
            res = subprocess.run(list_cmd, capture_output=True, text=True, timeout=15, cwd=benchmark_dir)
            print(f"List returncode: {res.returncode}; stdout_len={len(res.stdout or '')}")
            units = {}
            if res.returncode == 0 and res.stdout and res.stdout.strip():
                # Each non-empty line is a unit name
                unit_names = [l.strip() for l in res.stdout.splitlines() if l.strip()]
                print(f"Found {len(unit_names)} units via --bm_list for {benchmark_binary.name}")
                for unit in unit_names:
                    # run the benchmark for this unit to get its measured throughput (iters/s)
                    # try --benchmark with --bm_regex
                    bm_cmd = [str(benchmark_binary), "--benchmark", "--bm_regex", f"^{re.escape(unit)}$"]
                    try:
                        print(f"Probing unit '{unit}' with: {' '.join(bm_cmd)}")
                        r2 = subprocess.run(bm_cmd, capture_output=True, text=True, timeout=20, cwd=benchmark_dir)
                        print(f"Probe returncode={r2.returncode}; stdout_len={len(r2.stdout or '')}")
                        # try to parse throughput from output
                        parsed = self._extract_units_from_text(r2.stdout)
                        if unit in parsed:
                            units[unit] = parsed[unit]
                            continue
                        # fallback: try pattern matching for time/iter and iters/s columns
                        parsed_any = self._extract_units_from_text(r2.stdout)
                        if parsed_any:
                            # if only one entry, take its value
                            if len(parsed_any) == 1:
                                units[list(parsed_any.keys())[0]] = list(parsed_any.values())[0]
                                continue
                        # if we couldn't parse, set default throughput (very high) to get a small duration
                        units[unit] = 1e6
                    except subprocess.TimeoutExpired:
                        print(f"Probe for unit '{unit}' timed out")
                        units[unit] = 1e6
                return units
            # if --bm_list produced nothing, fall back to previous behavior
            print(f"--bm_list produced no output for {benchmark_binary.name}, falling back to --json/text discovery")
        except subprocess.TimeoutExpired:
            print(f"--bm_list timed out for {benchmark_binary.name}; falling back to --json/text discovery")

        # Fallback: try --json or plain run as before
        cmd = [str(benchmark_binary), "--json"]
        try:
            print(f"Running to discover units: {' '.join(cmd)} (cwd={benchmark_dir})")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=benchmark_dir)
            print(f"Discovery returncode: {result.returncode}; stdout_len={len(result.stdout or '')}; stderr_len={len(result.stderr or '')}")
            units = {}
            # Try JSON first
            try:
                data = json.loads(result.stdout)
                units = self._extract_units_from_json(data)
            except json.JSONDecodeError:
                units = self._extract_units_from_text(result.stdout)

            if not units:
                # helpful debug: show a truncated stdout/stderr to help diagnose format
                out_lines = (result.stdout or "").splitlines()
                err_lines = (result.stderr or "").splitlines()
                out_snip = "\n".join(out_lines[:80])
                err_snip = "\n".join(err_lines[:80])
                print(f"[discover units] No units parsed for {benchmark_binary.name}. stdout (truncated, first 80 lines):\n{out_snip}")
                if err_snip:
                    print(f"[discover units] stderr (truncated, first 80 lines):\n{err_snip}")

            return units
        except subprocess.TimeoutExpired as te:
            print(f"JSON discovery timed out for {benchmark_binary.name}; falling back to plain text parse: {te}")
            # Try running the binary without --json to get textual listing (short timeout)
            try:
                fallback = [str(benchmark_binary)]
                print(f"Running fallback discover command: {' '.join(fallback)}")
                result = subprocess.run(fallback, capture_output=True, text=True, timeout=15, cwd=benchmark_dir)
                print(f"Fallback returncode: {result.returncode}; stdout_len={len(result.stdout or '')}; stderr_len={len(result.stderr or '')}")
                units = self._extract_units_from_text(result.stdout)
                if not units:
                    print(f"Fallback parsing produced no units for {benchmark_binary.name}")
                return units
            except Exception as e2:
                print(f"Fallback discovery failed for {benchmark_binary.name}: {e2}")
                return {}
        except Exception as e:
            print(f"Error running benchmark to discover units: {e}")
            return {}

    def _extract_units_from_json(self, data: dict) -> Dict[str, float]:
        units = {}
        # Support dict or list outputs from various benchmark JSON formats
        if isinstance(data, dict):
            # Common Google benchmark style
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
        units = {}
        # Common Google Benchmark / folly textual output often has columns like:
        # <Name> <time/iter> <iters/s>
        # Example: "BinaryProtocol_write_Empty                                  5.96ns   167.85M"
        # First try to match that specific two-/three-column layout.
        col_pattern = re.compile(r'^\s*(?P<name>\S+)\s+(?P<time>[\d\.]+)\s*(?P<time_unit>ns|us|ms|s)?\s+(?P<iters>[\d\.,]+)\s*(?P<mult>[KMGTP]?)\b', re.IGNORECASE)

        # Fallback flexible pattern (older heuristic)
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

            # fallback
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
        
        # Check if already completed
        if self._is_completed(benchmark_name, unit_name):
            print(f"Skipping already completed: {benchmark_name} :: {unit_name}")
            return (True, "", 0.0)
        
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

        benchmark_cmd = [str(benchmark_binary), "--benchmark --bm_regex", f"^{re.escape(unit_name)}$"]

        print(f"Recording: {benchmark_name} :: {unit_name} for ~{duration:.3f}s -> perf sleep {perf_sleep}")
        print(f"Perf cmd: {' '.join(perf_cmd)}")
        print(f"Benchmark cmd: {' '.join(benchmark_cmd)} (cwd={benchmark_dir})")

        if dry_run:
            print("DRY-RUN: ", " ".join(perf_cmd))
            print("DRY-RUN: ", " ".join(benchmark_cmd))
            return (True, str(perf_data_file), 0.0)

        start = time.time()

        try:
            # Start perf record first

            perf_proc = subprocess.Popen(perf_cmd, cwd=benchmark_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Started perf PIDs: {getattr(perf_proc, 'pid', 'unknown')}")

            # Small delay to ensure perf record is up
            time.sleep(0.05)

            # Run benchmark; don't wait too long for it
            print("Launching benchmark process...")
            bench_proc = subprocess.run(
                benchmark_cmd, capture_output=True, text=True, cwd=benchmark_dir, timeout=perf_sleep + 5
            )
            print(f"Benchmark finished: returncode={bench_proc.returncode}; stdout_len={len(bench_proc.stdout or '')}; stderr_len={len(bench_proc.stderr or '')}")

            # Wait for perf to finish (it will finish when sleep finishes)
            print("Waiting for perf to finish...")
            perf_stdout, perf_stderr = perf_proc.communicate(timeout=perf_sleep + 10)
            print(f"Perf finished: stdout_len={len(perf_stdout or '')}; stderr_len={len(perf_stderr or '')}")

            elapsed = time.time() - start

            # Do not write per-unit bench_out_file on disk as requested; keep short debug output instead
            if bench_proc.stdout:
                print(f"[bench stdout] ({benchmark_name}::{unit_name}) {len(bench_proc.stdout)} bytes")
            if bench_proc.stderr:
                print(f"[bench stderr] ({benchmark_name}::{unit_name}) {len(bench_proc.stderr)} bytes")

            # Mark this work as completed
            self._mark_completed(benchmark_name, unit_name)
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

        total_work = 0
        skipped_count = 0
        failed_count = 0
        
        print(f"\n=== Starting with {len(self.completed_work)} previously completed work items ===")
        
        for name in sorted(selected.keys()):
            binary = selected[name]
            units = self.run_benchmark_get_units(binary)
            if not units:
                print(f"No units for {name}")
                continue

            for unit, thr in sorted(units.items()):
                total_work += 1
                if self._is_completed(name, unit):
                    skipped_count += 1
                    continue
                    
                ok, perf_path, elapsed = self.run_unit_record(binary, unit, thr, dry_run=dry_run)
                if not ok:
                    print(f"Failed: {name} :: {unit}")
                    failed_count += 1
        
        print(f"\n=== Summary ===")
        print(f"Total work items: {total_work}")
        print(f"Already completed: {skipped_count}")
        print(f"Failed in this run: {failed_count}")
        print(f"Completed in this run: {total_work - skipped_count - failed_count}")


def main():
    parser = argparse.ArgumentParser(description="Record Intel PT traces for folly microbenchmark units")
    parser.add_argument('--folly-test-dir', default='/mydata/DCPerf/benchmarks/wdl_bench/wdl_sources/folly/folly/test')
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
