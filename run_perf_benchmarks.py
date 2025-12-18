#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Script to run individual microbenchmark units with perf stat PMU measurements
# Measures each unit for its entire execution duration based on throughput

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

class BenchmarkUnitRunner:
    """Runs individual benchmark units with perf stat measurements."""
    
    def __init__(self, wdl_root: str, perf_events: str):
        self.wdl_root = Path(wdl_root)
        self.wdl_build = self.wdl_root / "wdl_build"
        self.perf_results_dir = self.wdl_root / "perf_results"
        self.perf_events = perf_events
        self.perf_results_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_units_from_json(self, json_file: Path) -> Set[str]:
        """Extract benchmark unit names from JSON output."""
        units = set()
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                if 'benchmarks' in data:
                    # Google Benchmark format
                    for bench in data['benchmarks']:
                        if 'name' in bench:
                            units.add(bench['name'])
                else:
                    # Direct key-value format (folly-style)
                    for key in data.keys():
                        if not key.startswith('%'):
                            units.add(key)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'name' in item:
                        units.add(item['name'])
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not parse JSON from {json_file}: {e}", file=sys.stderr)
        
        return units
    
    def extract_units_from_output(self, output_file: Path) -> Set[str]:
        """Extract benchmark units from text output using regex patterns."""
        units = set()
        
        try:
            with open(output_file, 'r') as f:
                content = f.read()
            
            # Try multiple patterns
            patterns = [
                r'^([a-zA-Z0-9_\-().]+):\s+\d+',  # name: number
                r'Running:\s+([a-zA-Z0-9_\-().]+)',  # Running: name
                r'bench\([^)]+\)',  # bench(...) format
                r'[a-zA-Z0-9_]+_memops\([^)]+\)',  # memops(...) format
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                units.update(matches)
        except IOError as e:
            print(f"Warning: Could not read output from {output_file}: {e}", file=sys.stderr)
        
        return units
    
    def run_benchmark_all_units(self, benchmark_name: str, config: str = "") -> List[str]:
        """Run a benchmark and extract all its units."""
        
        print(f"\n{'='*60}")
        print(f"Processing benchmark: {benchmark_name}")
        print(f"{'='*60}")
        
        benchmark_binary = self.wdl_build / benchmark_name
        if not benchmark_binary.exists():
            print(f"Error: Benchmark binary not found: {benchmark_binary}", file=sys.stderr)
            return []
        
        # Run benchmark to generate output
        output_file = self.wdl_root / f"out_{benchmark_name}.json"
        os.chdir(self.wdl_root)
        
        cmd = f"./{benchmark_name} {config}"
        print(f"Running: {cmd}")
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            with open(output_file, 'w') as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write(f"\nSTDERR:\n{result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"Warning: Benchmark {benchmark_name} timed out", file=sys.stderr)
            return []
        except Exception as e:
            print(f"Error running benchmark: {e}", file=sys.stderr)
            return []
        
        # Extract units
        units = self.extract_units_from_json(output_file)
        if not units:
            units = self.extract_units_from_output(output_file)
        
        if not units:
            print(f"Warning: No units extracted from {benchmark_name}")
            units = {"all"}
        
        print(f"Found {len(units)} benchmark units")
        return sorted(list(units))
    
    def run_unit_with_perf(self, benchmark_name: str, unit_name: str, 
                          config: str = "", duration: int = 30) -> Tuple[bool, str]:
        """Run a single benchmark unit with perf stat for the entire duration.
        
        The benchmark framework will run the unit repeatedly until the default
        benchmark time is reached (usually 100ms). Perf stat measures the entire
        time the benchmark runs.
        """
        
        benchmark_binary = self.wdl_build / benchmark_name
        os.chdir(self.wdl_root)
        
        # Sanitize filenames
        safe_unit_name = re.sub(r'[/\\:*?"<>|]', '_', unit_name)
        
        perf_output_file = self.perf_results_dir / f"{benchmark_name}_{safe_unit_name}.json"
        perf_text_file = self.perf_results_dir / f"{benchmark_name}_{safe_unit_name}.txt"
        
        # Build the benchmark command with pattern matching for specific unit
        benchmark_cmd = f"./{benchmark_name}"
        
        # Add pattern matching for specific unit if not already present
        if unit_name != "all" and "--bm_pattern" not in config and "--bm_regex" not in config:
            # Escape regex special characters
            escaped_unit = re.escape(unit_name)
            benchmark_cmd += f" --bm_pattern='{escaped_unit}'"
        
        if config:
            benchmark_cmd += f" {config}"
        
        # Note: We don't add --bm_min_time here because we want default timing
        # The benchmark will run for its standard duration (usually ~100ms per unit)
        # and perf stat will measure the entire execution
        
        # Build perf stat command to measure entire benchmark run
        # We use --sleep which measures for a specific duration, but we actually
        # want to let the benchmark run to completion. So we use a longer timeout.
        perf_cmd = [
            "sudo", "perf", "stat",
            "-a",  # all CPUs
            "-e", self.perf_events,
            "-o", str(perf_output_file),
            "--json"
        ]
        
        print(f"  Running: {unit_name:<70}", end=" ", flush=True)
        start_time = time.time()
        
        try:
            # Run perf stat wrapping the benchmark command
            # Let the benchmark run to completion naturally
            # Perf will measure until the benchmark process exits
            result = subprocess.run(
                perf_cmd + ["bash", "-c", benchmark_cmd],
                capture_output=True,
                text=True,
                timeout=duration * 5  # Allow generous timeout (5x the normal duration)
            )
            
            elapsed = time.time() - start_time
            
            # Save text output
            with open(perf_text_file, 'w') as f:
                f.write(f"Command: {benchmark_cmd}\n")
                f.write(f"Elapsed time: {elapsed:.2f}s\n")
                f.write(f"Perf output:\n{result.stdout}\n")
                if result.stderr:
                    f.write(f"\nSTDERR:\n{result.stderr}")
            
            print(f"✓ ({elapsed:.2f}s)")
            return (True, str(perf_output_file))
        
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            print(f"⏱ TIMEOUT ({elapsed:.2f}s)")
            return (False, "timeout")
        except PermissionError:
            print("✗ PERMISSION DENIED (sudo needed)")
            print(f"    Try: sudo {' '.join(perf_cmd)}")
            return (False, "permission_denied")
        except Exception as e:
            print(f"✗ ERROR: {e}")
            return (False, str(e))
    
    def parse_perf_results(self, json_file: Path) -> Dict:
        """Parse perf stat JSON results."""
        try:
            with open(json_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def run_all_benchmarks(self, benchmark_name: Optional[str] = None, 
                          duration: int = 30, config_overrides: Optional[Dict] = None):
        """Run all benchmarks or a specific one with perf measurements."""
        
        # Default configurations for each benchmark
        default_config = {
            'random_benchmark': "--bm_regex=xoshiro --json",
            'memcpy_benchmark': "--json",
            'memset_benchmark': "--json",
            'hash_hash_benchmark': "--bm_regex=RapidHash --json",
            'hash_checksum_benchmark': "--json",
            'synchronization_lifo_sem_bench': "--bm_min_iters=1000000 --json",
            'synchronization_small_locks_benchmark': "--bm_min_iters=1000000 --bm_regex=folly_RWSpinlock --json",
            'container_hash_maps_bench': '--bm_regex="f14(vec)|(val)" --json',
            'ProtocolBench': '--bm_regex="(^Binary)|(^Compact)Protocol" --json',
            'VarintUtilsBench': "--json",
            'bench-memcmp': "",
            'benchsleef128': "--benchmark_format=json",
            'benchsleef256': "--benchmark_format=json",
            'benchsleef512': "--benchmark_format=json",
        }
        
        if config_overrides:
            default_config.update(config_overrides)
        
        benchmarks_to_run = []
        if benchmark_name:
            if benchmark_name in default_config or (self.wdl_build / benchmark_name).exists():
                benchmarks_to_run = [benchmark_name]
            else:
                print(f"Error: Benchmark '{benchmark_name}' not found", file=sys.stderr)
                return
        else:
            # Find all available benchmarks
            for bench in default_config:
                if (self.wdl_build / bench).exists():
                    benchmarks_to_run.append(bench)
        
        summary = {}
        start_time = datetime.now()
        
        for benchmark in benchmarks_to_run:
            config = default_config.get(benchmark, "")
            
            # Run benchmark and get units
            units = self.run_benchmark_all_units(benchmark, config)
            
            if not units:
                continue
            
            benchmark_results = {
                'units_count': len(units),
                'units': [],
                'failed': []
            }
            
            # Run each unit with perf
            for unit in units:
                success, result_file = self.run_unit_with_perf(benchmark, unit, config, duration)
                
                if success:
                    benchmark_results['units'].append({
                        'name': unit,
                        'perf_result': result_file
                    })
                else:
                    benchmark_results['failed'].append(unit)
            
            summary[benchmark] = benchmark_results
        
        # Print summary
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total elapsed time: {elapsed:.2f}s")
        print(f"Results directory: {self.perf_results_dir}")
        print()
        
        for benchmark, results in summary.items():
            print(f"{benchmark}:")
            print(f"  Units processed: {results['units_count']}")
            print(f"  Successful: {len(results['units'])}")
            if results['failed']:
                print(f"  Failed: {len(results['failed'])}")
                for failed_unit in results['failed'][:5]:  # Show first 5
                    print(f"    - {failed_unit}")
                if len(results['failed']) > 5:
                    print(f"    ... and {len(results['failed']) - 5} more")
        
        print()
        print(f"Perf results saved to: {self.perf_results_dir}")
        print("Run 'python3 analyze_perf_results.py' to analyze results")

def main():
    parser = argparse.ArgumentParser(
        description="Run individual microbenchmark units with perf stat PMU measurements"
    )
    parser.add_argument(
        '--name',
        help='Specific benchmark to run (e.g., memcpy_benchmark)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=30,
        help='Duration in seconds for each benchmark unit (default: 30)'
    )
    parser.add_argument(
        '--events',
        default='cycles,instructions,L1-icache-load-misses,iTLB-loads,iTLB-load-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,LLC-load-misses,branch-load-misses,branch-misses,r2424',
        help='PMU events to measure'
    )
    
    args = parser.parse_args()
    
    # Get WDL root
    script_dir = Path(__file__).parent.absolute()
    wdl_root = script_dir
    
    runner = BenchmarkUnitRunner(str(wdl_root), args.events)
    runner.run_all_benchmarks(
        benchmark_name=args.name,
        duration=args.duration
    )

if __name__ == '__main__':
    main()
