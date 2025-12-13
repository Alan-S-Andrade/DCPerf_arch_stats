#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Advanced script to run folly microbenchmark units with perf stat
# Automatically discovers benchmarks and measures each unit for its entire duration

import json
import subprocess
import os
import sys
import re
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set, Tuple, Optional
import argparse
import tempfile

class FollyBenchmarkPerfRunner:
    """Run folly benchmark units with perf stat measurements."""
    
    def __init__(self, folly_test_dir: str, output_dir: str, perf_events: str):
        self.folly_test_dir = Path(folly_test_dir)
        self.output_dir = Path(output_dir)
        self.perf_events = perf_events
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.benchmarks = {}  # benchmark_name -> binary_path
        self.benchmark_configs = {}  # benchmark_name -> config_string
        self.discover_benchmarks()
    
    def discover_benchmarks(self):
        """Discover all benchmark binaries in the wdl_bench directory."""
        
        print(f"Scanning for benchmark binaries...")
        
        # Try multiple paths to find benchmarks
        possible_paths = [
            Path("/users/alanuiuc/DCPerf/benchmarks/wdl_bench"),
            Path("/users/alanuiuc/DCPerf/benchmarks/wdl_bench/wdl_build/build/folly"),
            Path("/users/alanuiuc/DCPerf/benchmarks/wdl_bench/wdl_build"),
            self.folly_test_dir.parent.parent.parent.parent / "wdl_build",
            Path.cwd() / "wdl_build",
            Path.cwd().parent / "wdl_build",
        ]
        
        wdl_build_base = None
        for path in possible_paths:
            if path.exists():
                # Check if this path has actual benchmark binaries
                test_files = list(path.glob("memcpy_benchmark")) + list(path.glob("*benchmark"))
                if test_files:
                    wdl_build_base = path
                    print(f"  Found benchmarks at: {wdl_build_base}")
                    break
        
        if wdl_build_base is None:
            print(f"  Warning: No benchmark directory found")
            print(f"  Searched in:")
            for p in possible_paths:
                print(f"    - {p}")
            return
        
        found_count = 0
        for binary in sorted(wdl_build_base.glob("*")):
            if binary.is_file() and os.access(binary, os.X_OK):
                bench_name = binary.name
                # Filter to likely benchmarks
                if any(p in bench_name.lower() for p in ['benchmark', 'bench', 'perf']):
                    self.benchmarks[bench_name] = binary
                    found_count += 1
                    print(f"  Found: {bench_name}")
        
        if found_count == 0:
            print(f"  No executable benchmark binaries found in {wdl_build_base}")
            print(f"  Check with: ls -la {wdl_build_base}")
    
    def run_benchmark_get_units(self, benchmark_name: str) -> Dict[str, float]:
        """Run benchmark and extract unit names with their estimated throughput.
        
        Returns dict mapping unit_name -> throughput (iters/sec)
        """
        
        if benchmark_name not in self.benchmarks:
            return {}
        
        benchmark_binary = self.benchmarks[benchmark_name]
        benchmark_dir = benchmark_binary.parent
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_output = Path(tmpdir) / "output.json"
            
            cmd = [str(benchmark_binary), "--json"]
            
            print(f"\n{'='*70}")
            print(f"Extracting units from: {benchmark_name}")
            print(f"{'='*70}")
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=benchmark_dir  # Run from benchmark directory
                )
                
                units = {}
                
                # Try to parse JSON output
                try:
                    data = json.loads(result.stdout)
                    units = self._extract_units_from_json(data)
                except json.JSONDecodeError:
                    # Try alternative parsing
                    units = self._extract_units_from_text(result.stdout)
                
                print(f"Found {len(units)} benchmark units")
                return units
            
            except subprocess.TimeoutExpired:
                print(f"  Warning: Benchmark timed out")
                return {}
            except Exception as e:
                print(f"  Error: {e}")
                return {}
    
    def _extract_units_from_json(self, data: dict) -> Dict[str, float]:
        """Extract unit names and throughput from JSON data."""
        
        units = {}
        
        if isinstance(data, dict):
            if 'benchmarks' in data:
                # Google Benchmark format
                for bench in data['benchmarks']:
                    if 'name' in bench:
                        name = bench['name']
                        # Try to extract throughput (items_per_second or similar)
                        throughput = bench.get('items_per_second', bench.get('iterations_per_second', 1e6))
                        units[name] = float(throughput) if throughput else 1e6
            else:
                # Direct key-value format (folly-style)
                for key, value in data.items():
                    if not key.startswith('%') and isinstance(value, (int, float)):
                        units[key] = float(value)
        
        return units
    
    def _extract_units_from_text(self, text: str) -> Dict[str, float]:
        """Extract unit names from text output using regex."""
        
        units = {}
        
        # Pattern to match benchmark results like:
        # bench(0_to_1024_COLD_folly)                                     2.21ns   452.91M
        pattern = r'([a-zA-Z0-9_\-().]+)\s+[\d.]+\s+[a-z]+\s+([\d.]+)([KMGT]?)'
        
        for match in re.finditer(pattern, text, re.MULTILINE):
            unit_name = match.group(1)
            throughput_val = float(match.group(2))
            multiplier = match.group(3)
            
            # Convert to per-second
            multipliers = {'K': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12}
            throughput = throughput_val * multipliers.get(multiplier, 1)
            
            if unit_name and throughput > 0:
                units[unit_name] = throughput
        
        return units
    
    def calculate_measurement_duration(self, throughput: float, target_iterations: int = 100000) -> float:
        """Calculate measurement duration based on throughput.
        
        Ensures we capture at least target_iterations worth of data.
        throughput: iterations/second
        """
        
        if throughput <= 0:
            return 0.5  # Default 500ms
        
        # Duration = (target iterations / throughput)
        duration = target_iterations / throughput
        
        # Clamp to reasonable range (50ms - 60 seconds)
        duration = max(0.05, min(60.0, duration))
        
        return duration
    
    def run_unit_with_perf(self, benchmark_binary: Path, unit_name: str,
                          throughput: float = 0) -> Tuple[bool, str, float]:
        """Run a single benchmark unit with perf stat for appropriate duration.
        
        Returns (success, output_file, elapsed_time)
        """
        
        benchmark_name = benchmark_binary.name
        benchmark_dir = benchmark_binary.parent
        safe_unit_name = re.sub(r'[/\\:*?"<>|]', '_', unit_name)
        
        perf_output_file = self.output_dir / f"{benchmark_name}_{safe_unit_name}_perf.txt"
        perf_json_file = self.output_dir / f"{benchmark_name}_{safe_unit_name}_perf.json"
        
        # Build benchmark command with unit pattern
        benchmark_cmd = [str(benchmark_binary), "--bm_pattern", f"^{re.escape(unit_name)}$"]
        
        # Build perf stat command
        perf_cmd = [
            "sudo", "perf", "stat",
            "-a",  # All CPUs
            "-e", self.perf_events,
        ]
        
        # Estimate measurement duration
        duration = self.calculate_measurement_duration(throughput, target_iterations=50000)
        timeout = max(5, duration * 3)  # 3x safety margin
        
        # Print progress
        throughput_str = f"{throughput/1e6:.1f}M" if throughput >= 1e6 else f"{throughput/1e3:.1f}K"
        duration_str = f"{duration*1000:.0f}ms" if duration < 1 else f"{duration:.2f}s"
        
        print(f"  {unit_name:<65} [{throughput_str:>8} iter/s, ~{duration_str:>6}]", end=" ", flush=True)
        
        start_time = time.time()
        
        try:
            # Run with perf stat from the benchmark directory
            result = subprocess.run(
                perf_cmd + benchmark_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=benchmark_dir  # Run from benchmark directory
            )
            
            elapsed = time.time() - start_time
            
            # Save text output (perf stat output goes to stderr)
            perf_output = result.stderr if result.stderr else result.stdout
            
            with open(perf_output_file, 'w') as f:
                f.write(f"Benchmark: {benchmark_name}\n")
                f.write(f"Unit: {unit_name}\n")
                f.write(f"Throughput estimate: {throughput:.2e} iters/s\n")
                f.write(f"Target duration: {duration:.3f}s\n")
                f.write(f"Actual elapsed: {elapsed:.3f}s\n")
                f.write(f"\nCommand: {' '.join(benchmark_cmd)}\n\n")
                f.write(f"Perf output:\n{perf_output}\n")
                if result.stdout:
                    f.write(f"\nBenchmark output:\n{result.stdout}\n")
            
            print(f"✓ ({elapsed:.2f}s)")
            return (True, str(perf_output_file), elapsed)
        
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            print(f"⏱ ({elapsed:.2f}s)")
            return (False, "", elapsed)
        except PermissionError:
            print("✗ (need sudo)")
            return (False, "", 0)
        except Exception as e:
            print(f"✗ ({e})")
            return (False, "", 0)
    
    def run_all_benchmarks(self, benchmark_filter: Optional[str] = None):
        """Run all discovered benchmarks with perf measurements."""
        
        if not self.benchmarks:
            print("No benchmarks found!")
            return
        
        # Filter benchmarks if requested
        benchmarks_to_run = self.benchmarks
        if benchmark_filter:
            benchmarks_to_run = {
                k: v for k, v in self.benchmarks.items()
                if benchmark_filter.lower() in k.lower()
            }
        
        total_start = time.time()
        summary = {}
        
        for benchmark_name in sorted(benchmarks_to_run.keys()):
            benchmark_binary = benchmarks_to_run[benchmark_name]
            
            # Extract units and throughput
            units = self.run_benchmark_get_units(benchmark_name)
            
            if not units:
                print(f"  (no units found or benchmark failed)")
                continue
            
            benchmark_results = {
                'total_units': len(units),
                'successful': 0,
                'failed': 0,
                'total_time': 0
            }
            
            # Run each unit with perf
            for unit_name in sorted(units.keys()):
                success, output_file, elapsed = self.run_unit_with_perf(
                    benchmark_binary,
                    unit_name,
                    units[unit_name]
                )
                
                benchmark_results['total_time'] += elapsed
                if success:
                    benchmark_results['successful'] += 1
                else:
                    benchmark_results['failed'] += 1
            
            summary[benchmark_name] = benchmark_results
        
        # Print final summary
        total_elapsed = time.time() - total_start
        
        print(f"\n{'='*70}")
        print("MEASUREMENT SUMMARY")
        print(f"{'='*70}\n")
        
        total_units = 0
        total_success = 0
        total_failed = 0
        
        for benchmark_name, results in summary.items():
            print(f"{benchmark_name}:")
            print(f"  Total units: {results['total_units']}")
            print(f"  Successful: {results['successful']}")
            if results['failed']:
                print(f"  Failed: {results['failed']}")
            print(f"  Time: {results['total_time']:.1f}s")
            print()
            
            total_units += results['total_units']
            total_success += results['successful']
            total_failed += results['failed']
        
        print(f"{'='*70}")
        print(f"Total units measured: {total_success}/{total_units}")
        if total_failed:
            print(f"Failed: {total_failed}")
        print(f"Total elapsed time: {total_elapsed/60:.1f} minutes ({total_elapsed:.0f}s)")
        print(f"Results directory: {self.output_dir}")
        print(f"{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Run folly benchmark units with perf stat PMU measurements"
    )
    parser.add_argument(
        '--folly-test-dir',
        default='/users/alanuiuc/DCPerf/benchmarks/wdl_bench/wdl_sources/folly/folly/test',
        help='Path to folly test directory'
    )
    parser.add_argument(
        '--output-dir',
        default='./perf_results_detailed',
        help='Output directory for perf results'
    )
    parser.add_argument(
        '--benchmark',
        help='Specific benchmark to run (can be partial match)'
    )
    parser.add_argument(
        '--events',
        default='cycles,instructions,L1-icache-load-misses,iTLB-load-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,LLC-load-misses,branch-load-misses,branch-misses,r2424',
        help='PMU events to measure'
    )
    
    args = parser.parse_args()
    
    # Make output directory absolute if it's relative
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    
    runner = FollyBenchmarkPerfRunner(
        args.folly_test_dir,
        str(output_dir),
        args.events
    )
    
    runner.run_all_benchmarks(args.benchmark)

if __name__ == '__main__':
    main()
