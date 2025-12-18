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
    
    def __init__(self, folly_test_dir: str, output_dir: str, perf_events: str, use_probe_cache: bool = True):
        self.folly_test_dir = Path(folly_test_dir)
        self.output_dir = Path(output_dir)
        self.perf_events = perf_events
        self.use_probe_cache = use_probe_cache
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file for probe results
        self.probe_cache_file = self.output_dir / "probe_cache.json"
        self.probe_cache = {}  # benchmark_name -> {unit_name -> throughput}
        
        # Load existing probe cache
        if self.use_probe_cache and self.probe_cache_file.exists():
            self._load_probe_cache()
        
        self.benchmarks = {}  # benchmark_name -> binary_path
        self.benchmark_configs = {}  # benchmark_name -> config_string
        self.discover_benchmarks()
    
    def _load_probe_cache(self):
        """Load probe cache from disk."""
        try:
            with open(self.probe_cache_file, 'r') as f:
                self.probe_cache = json.load(f)
            print(f"Loaded probe cache with {sum(len(v) for v in self.probe_cache.values())} units")
        except Exception as e:
            print(f"Warning: Failed to load probe cache: {e}")
            self.probe_cache = {}
    
    def _save_probe_cache(self):
        """Save probe cache to disk."""
        try:
            with open(self.probe_cache_file, 'w') as f:
                json.dump(self.probe_cache, f, indent=2)
            total_units = sum(len(v) for v in self.probe_cache.values())
            print(f"Saved probe cache with {total_units} units")
        except Exception as e:
            print(f"Warning: Failed to save probe cache: {e}")
    
    def _create_unit_cache_entry(self, throughput: float) -> Dict[str, float]:
        """Create a cache entry for a unit with throughput and estimated duration."""
        duration = self.calculate_measurement_duration(throughput, target_iterations=50000)
        return {
            'throughput': throughput,
            'duration': duration
        }
    
    def _get_throughput_from_cache_entry(self, entry) -> float:
        """Extract throughput from cache entry, handling both old and new formats."""
        if isinstance(entry, dict):
            return entry.get('throughput', 1e6)
        else:
            # Handle old format (just a float)
            return float(entry) if entry else 1e6
    
    def _get_duration_from_cache_entry(self, entry) -> float:
        """Extract duration from cache entry, or calculate if not present."""
        if isinstance(entry, dict):
            if 'duration' in entry:
                return entry['duration']
            # If we have throughput, calculate duration
            throughput = entry.get('throughput', 1e6)
            return self.calculate_measurement_duration(throughput, target_iterations=50000)
        else:
            # Old format - calculate from throughput
            throughput = float(entry) if entry else 1e6
            return self.calculate_measurement_duration(throughput, target_iterations=50000)
    
    def discover_benchmarks(self):
        """Discover all benchmark binaries in the wdl_bench directory."""
        print(f"Scanning for benchmark binaries...")

        possible_paths = [
            Path("/myd/DC2/DCPerf/benchmarks/r_wdl_bench"),
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
    
    def run_benchmark_get_units(self, benchmark_name: str) -> Dict[str, float]:
        """Run benchmark and extract unit names with their estimated throughput.
        
        Returns dict mapping unit_name -> throughput (iters/sec)
        Checks cache first if available.
        """
        
        if benchmark_name not in self.benchmarks:
            return {}
        
        # Check if we have cached results for this benchmark
        if self.use_probe_cache and benchmark_name in self.probe_cache:
            cached_units = self.probe_cache[benchmark_name]
            units = {}
            for unit_name, entry in cached_units.items():
                units[unit_name] = self._get_throughput_from_cache_entry(entry)
            print(f"\n{'='*70}")
            print(f"Extracting units from: {benchmark_name} (from cache)")
            print(f"{'='*70}")
            print(f"  Found {len(units)} units via cache")
            return units

        benchmark_binary = self.benchmarks[benchmark_name]
        benchmark_dir = benchmark_binary.parent

        print(f"\n{'='*70}")
        print(f"Extracting units from: {benchmark_name}")
        print(f"{'='*70}")

        # 1) Prefer --bm_list if available, then probe each unit with --benchmark --bm_regex '^unit$'
        try:
            list_cmd = [str(benchmark_binary), "--bm_list"]
            list_res = subprocess.run(
                list_cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=benchmark_dir
            )

            stdout = (list_res.stdout or "").strip()
            if stdout:
                # Parse unit names from --bm_list output
                # Handles both simple list and formatted table output with timing data
                unit_lines = []
                for line in stdout.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    # Skip header/separator lines
                    if '=====' in line or '.....' in line or 'Test name' in line or 'threads' in line:
                        continue
                    # Extract unit name (everything before the first timing value with "ns")
                    # Pattern: unit_name ... XXX ns YYY ns ZZZ ns
                    if 'ns' in line:
                        # Extract unit name by finding where the numbers start
                        match = re.match(r'^([^0-9]+?)\s+\d+\s*ns', line)
                        if match:
                            unit_name = match.group(1).strip()
                            if unit_name:
                                unit_lines.append(unit_name)
                    else:
                        # Simple list format - just unit names, no timing data
                        unit_lines.append(line)
                
                unit_lines = [l for l in unit_lines if l]  # Filter empty strings
                if unit_lines:
                    print(f"  Found {len(unit_lines)} units via --bm_list")
                    units: Dict[str, float] = {}
                    cache_entries: Dict[str, Dict[str, float]] = {}
                    for unit in unit_lines:
                        print(f"  Probing unit: {unit}")
                        probe_cmd = [str(benchmark_binary), "--benchmark", "--bm_regex", f'^{re.escape(unit)}$']
                        try:
                            p = subprocess.run(
                                probe_cmd,
                                capture_output=True,
                                text=True,
                                timeout=30,
                                cwd=benchmark_dir
                            )

                            # Try to parse textual output for throughput
                            parsed = self._extract_units_from_text(p.stdout)
                            if parsed:
                                # Prefer a parsed key that matches the unit name
                                matched = None
                                for k in parsed.keys():
                                    if unit in k or k in unit:
                                        matched = k
                                        break
                                if matched is None:
                                    matched = next(iter(parsed.keys()))
                                throughput = parsed.get(matched, 1e6)
                                units[unit] = throughput
                                cache_entries[unit] = self._create_unit_cache_entry(throughput)
                            else:
                                # Fallback estimate
                                units[unit] = 1e6
                                cache_entries[unit] = self._create_unit_cache_entry(1e6)
                        except subprocess.TimeoutExpired:
                            print(f"    probe timed out for unit: {unit}")
                            units[unit] = 1e6
                            cache_entries[unit] = self._create_unit_cache_entry(1e6)
                        except Exception as e:
                            print(f"    probe error for unit {unit}: {e}")
                            units[unit] = 1e6
                            cache_entries[unit] = self._create_unit_cache_entry(1e6)

                    print(f"  Probed {len(units)} units via --bm_list")
                    # Cache the results with duration info
                    if self.use_probe_cache:
                        self.probe_cache[benchmark_name] = cache_entries
                        self._save_probe_cache()
                    return units

        except subprocess.TimeoutExpired:
            print("  --bm_list timed out; falling back")
        except Exception as e:
            print(f"  --bm_list failed: {e}; falling back")

        # 2) Fallback: try --json (older behavior)
        try:
            cmd = [str(benchmark_binary), "--json"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=benchmark_dir
            )

            units = {}
            try:
                data = json.loads(result.stdout)
                units = self._extract_units_from_json(data)
            except json.JSONDecodeError:
                units = self._extract_units_from_text(result.stdout)

            print(f"Found {len(units)} benchmark units (fallback)")
            # Cache the results with duration info
            if self.use_probe_cache:
                cache_entries = {unit: self._create_unit_cache_entry(throughput) 
                                 for unit, throughput in units.items()}
                self.probe_cache[benchmark_name] = cache_entries
                self._save_probe_cache()
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
                          throughput: float = 0, cached_duration: Optional[float] = None) -> Tuple[bool, str, float]:
        """Run a single benchmark unit with perf stat for appropriate duration.
        
        Args:
            benchmark_binary: Path to benchmark binary
            unit_name: Name of unit to benchmark
            throughput: Throughput in iters/sec
            cached_duration: Optional cached duration from previous probe
        
        Returns (success, output_file, elapsed_time)
        """
        
        benchmark_name = benchmark_binary.name
        benchmark_dir = benchmark_binary.parent
        safe_unit_name = re.sub(r'[/\\:*?"<>|]', '_', unit_name)
        
        perf_output_file = self.output_dir / f"{benchmark_name}_{safe_unit_name}_perf.txt"
        perf_json_file = self.output_dir / f"{benchmark_name}_{safe_unit_name}_perf.json"
        
        # Build benchmark command with unit pattern (same format as probing stage)
        benchmark_cmd = [str(benchmark_binary), "--benchmark", "--bm_regex", f"^{re.escape(unit_name)}$"]
        
        # Build perf stat command
        perf_cmd = [
            "sudo", "perf", "stat",
            "-a",  # All CPUs
            "-e", self.perf_events,
        ]
        
        # Use cached duration if available, otherwise calculate
        if cached_duration is not None:
            duration = cached_duration
        else:
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
    
    def _cleanup_old_perf_files(self):
        """Clean up old perf result files from output directory."""
        if not self.output_dir.exists():
            return
        
        old_perf_files = list(self.output_dir.glob("*_perf.txt")) + list(self.output_dir.glob("*_perf.json"))
        if old_perf_files:
            print(f"Cleaning up {len(old_perf_files)} old perf result files...")
            for file in old_perf_files:
                try:
                    file.unlink()
                except Exception as e:
                    print(f"  Warning: Failed to delete {file}: {e}")
    
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
            
            # Extract cached durations if available
            cached_durations = {}
            if self.use_probe_cache and benchmark_name in self.probe_cache:
                cached_entries = self.probe_cache[benchmark_name]
                for unit_name, entry in cached_entries.items():
                    cached_durations[unit_name] = self._get_duration_from_cache_entry(entry)
            
            benchmark_results = {
                'total_units': len(units),
                'successful': 0,
                'failed': 0,
                'total_time': 0
            }
            
            # Run each unit with perf
            for unit_name in sorted(units.keys()):
                # Check if perf output file already exists
                safe_unit_name = re.sub(r'[/\\:*?"<>|]', '_', unit_name)
                perf_output_file = self.output_dir / f"{benchmark_name}_{safe_unit_name}_perf.txt"
                
                if perf_output_file.exists():
                    print(f"  {unit_name:<65} [skipped (exists)]")
                    benchmark_results['successful'] += 1
                    continue
                
                cached_duration = cached_durations.get(unit_name)
                success, output_file, elapsed = self.run_unit_with_perf(
                    benchmark_binary,
                    unit_name,
                    units[unit_name],
                    cached_duration=cached_duration
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
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable probe cache, force re-probing of all benchmarks'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear probe cache before running'
    )
    
    args = parser.parse_args()
    
    # Make output directory absolute if it's relative
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    
    # Handle cache clearing
    if args.clear_cache:
        cache_file = output_dir / "probe_cache.json"
        if cache_file.exists():
            cache_file.unlink()
            print(f"Cleared probe cache: {cache_file}")
    
    runner = FollyBenchmarkPerfRunner(
        args.folly_test_dir,
        str(output_dir),
        args.events,
        use_probe_cache=not args.no_cache
    )
    
    runner.run_all_benchmarks(args.benchmark)

if __name__ == '__main__':
    main()
