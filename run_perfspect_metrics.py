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
import sys
import tempfile


def _resolve_feature_window(feature_warmup_s: Optional[float], feature_window_s: Optional[float]) -> tuple:
    """Resolve effective warmup/window from explicit args, then env vars, then defaults."""
    warmup_val = feature_warmup_s
    window_val = feature_window_s

    if warmup_val is None:
        try:
            warmup_val = float(os.environ.get("FEATURE_WARMUP_S", "0") or 0)
        except ValueError:
            warmup_val = 0.0
    if window_val is None:
        try:
            window_val = float(os.environ.get("FEATURE_WINDOW_S", "0") or 0)
        except ValueError:
            window_val = 0.0

    warmup_val = max(0.0, float(warmup_val))
    window_val = max(0.0, float(window_val))
    return warmup_val, window_val


def _build_windowed_cmd(cmd_list: list, warmup_s: float, window_s: float) -> list:
    """Wrap command with optional warmup sleep and timeout window."""
    if warmup_s <= 0 and window_s <= 0:
        return cmd_list

    quoted_cmd = " ".join(shlex.quote(tok) for tok in cmd_list)
    run_part = f"timeout {window_s:g} {quoted_cmd}" if window_s > 0 else quoted_cmd

    if warmup_s > 0:
        shell_cmd = f"sleep {warmup_s:g}; exec {run_part}"
    else:
        shell_cmd = f"exec {run_part}"

    return ["bash", "-lc", shell_cmd]


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
            res = subprocess.run(list_cmd, capture_output=True, text=True, cwd=benchmark_dir)
            if res.returncode == 0 and res.stdout and res.stdout.strip():
                unit_names = [l.strip() for l in res.stdout.splitlines() if l.strip()]
                units = {}
                for unit in unit_names:
                    units[unit] = 1e6  # Default throughput estimate
                return units, cached_times

        # For folly benchmarks, extract units
        # Primary path: use --bm_list to enumerate unit names
        list_cmd = [str(benchmark_binary), "--bm_list"]
        print(f"  Running to list units: {' '.join(list_cmd)}")
        res = subprocess.run(list_cmd, capture_output=True, text=True, cwd=benchmark_dir)
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
                    r2 = subprocess.run(bm_cmd, capture_output=True, text=True, cwd=benchmark_dir)
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


def run_single_cmd_perfspect(
    cmd_list,
    perfspect_bin='perfspect',
    extra_args=None,
    output_dir=None,
    feature_warmup_s: Optional[float] = None,
    feature_window_s: Optional[float] = None,
):
    """Run perfspect metrics on a single arbitrary command and return parsed TMA metrics.
    
    Args:
        cmd_list: List of strings representing the command to run.
        perfspect_bin: Path to the perfspect binary.
        extra_args: Optional list of extra arguments for perfspect.
        output_dir: Optional directory for perfspect output files.
    
    Returns:
        Dict of TMA metric_name -> float_value, e.g.:
          {"TMA_Frontend_Bound(%)": 21.85, "TMA_..Fetch_Latency(%)": 2.64, ...}
    """
    import csv
    import shutil
    
    if output_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="perfspect_"))
    else:
        work_dir = Path(output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    perfspect_out_dir = work_dir / "perfspect_output"
    perfspect_out_dir.mkdir(parents=True, exist_ok=True)
    
    if extra_args is None:
        extra_args = []

    warmup_s, window_s = _resolve_feature_window(feature_warmup_s, feature_window_s)
    effective_cmd = _build_windowed_cmd(cmd_list, warmup_s, window_s)
    
    # Build perfspect command
    perfspect_cmd = [perfspect_bin, "metrics"] + extra_args + \
                    ["--output", str(perfspect_out_dir), "--"] + effective_cmd
    
    print(f"\n{'='*70}")
    print(f"[perfspect] Running: {' '.join(cmd_list)}")
    if warmup_s > 0 or window_s > 0:
        print(f"[perfspect] Window mode: warmup={warmup_s:.1f}s, window={window_s:.1f}s")
    print(f"[perfspect] Full command: {' '.join(perfspect_cmd)}")
    print(f"{'='*70}")
    
    try:
        start = time.time()
        result = subprocess.run(perfspect_cmd, capture_output=True, text=True)
        elapsed = time.time() - start
        
        print(f"[perfspect] Completed in {elapsed:.2f}s (returncode={result.returncode})")
        
        # Save raw output
        raw_log = work_dir / "perfspect_raw.log"
        with open(raw_log, 'w') as f:
            f.write(f"Command: {' '.join(cmd_list)}\n")
            f.write(f"Elapsed: {elapsed:.3f}s\n")
            f.write(f"Returncode: {result.returncode}\n\n")
            f.write("=== STDOUT ===\n")
            f.write(result.stdout or '')
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr or '')
        
    except Exception as e:
        print(f"[perfspect] Error running perfspect: {e}")
        return {}
    
    # Find the summary CSV in the output directory
    summary_files = list(perfspect_out_dir.glob("*_summary.csv")) + \
                    list(perfspect_out_dir.glob("*_metrics_summary.csv"))
    
    if not summary_files:
        # Also check for regular metrics CSV
        summary_files = list(perfspect_out_dir.glob("*.csv"))
    
    if not summary_files:
        print(f"[perfspect] No CSV output files found in {perfspect_out_dir}")
        if result.stdout:
            print(f"[perfspect] stdout: {result.stdout[:500]}")
        if result.stderr:
            print(f"[perfspect] stderr: {result.stderr[:500]}")
        return {}
    
    # Parse the summary CSV for TMA metrics
    # Format: metric,mean,min,max,stddev
    # We want TMA_* rows, taking the 'mean' column
    metrics = {}
    
    for csv_file in summary_files:
        print(f"[perfspect] Parsing: {csv_file.name}")
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metric_name = row.get('metric', '').strip()
                    if not metric_name:
                        continue
                    
                    # Get the mean value (prefer 'mean', fall back to 'value' or first numeric column)
                    val_str = row.get('mean', row.get('value', '')).strip()
                    if not val_str or val_str == 'NaN':
                        continue
                    
                    try:
                        val = float(val_str)
                    except ValueError:
                        continue
                    
                    # Only include TMA metrics
                    if metric_name.startswith('TMA_'):
                        metrics[metric_name] = round(val, 6)
        except Exception as e:
            print(f"[perfspect] Error parsing {csv_file.name}: {e}")
    
    print(f"[perfspect] Collected {len(metrics)} TMA metrics")
    return metrics


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
    p.add_argument(
        '--cmd',
        nargs=argparse.REMAINDER,
        default=None,
        help='Run perfspect metrics on a single command. Everything after --cmd is treated as the command. '
             'Example: --cmd sysbench cpu --cpu-max-prime=20000 --threads=8 --time=8 run'
    )
    p.add_argument('--feature-warmup-seconds', type=float, default=0.0,
                   help='Delay before starting command in single-command mode (default: 0)')
    p.add_argument('--feature-window-seconds', type=float, default=0.0,
                   help='Limit command runtime to N seconds in single-command mode (default: 0)')
    p.add_argument('--cloud-mode', action='store_true',
                   help='Convenience mode for cloud workloads: warmup=30s, window=10s')

    args = p.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    
    # Single-command mode
    if args.cmd:
        cmd_list = args.cmd
        # Strip leading '--' if present (argparse REMAINDER quirk)
        if cmd_list and cmd_list[0] == '--':
            cmd_list = cmd_list[1:]
        if not cmd_list:
            print("Error: --cmd requires a command to run")
            import sys
            sys.exit(1)

        warmup_s = float(args.feature_warmup_seconds)
        window_s = float(args.feature_window_seconds)
        if args.cloud_mode:
            warmup_s = 30.0
            window_s = 10.0
        if warmup_s < 0 or window_s < 0:
            print("Error: --feature-warmup-seconds and --feature-window-seconds must be >= 0")
            sys.exit(1)

        os.environ["FEATURE_WARMUP_S"] = str(warmup_s)
        os.environ["FEATURE_WINDOW_S"] = str(window_s)
        
        extra_args = shlex.split(args.perfspect_args) if args.perfspect_args else []
        metrics = run_single_cmd_perfspect(
            cmd_list,
            args.perfspect_bin,
            extra_args,
            str(out_dir),
            feature_warmup_s=warmup_s,
            feature_window_s=window_s,
        )
        
        # Save metrics to JSON
        out_dir.mkdir(parents=True, exist_ok=True)
        json_file = out_dir / "perfspect_metrics.json"
        with open(json_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"[perfspect] Metrics saved to {json_file}")
        return
    
    # Original batch mode
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
    # Don't add --output here; we'll add it per microbenchmark

    summary = {}
    total_start = time.time()

    for bench_name in sorted(benchmarks.keys()):
        bench_path = benchmarks[bench_name]
        print(f"\n== {bench_name} ==")

        # Create directory for this binary
        bench_dir = out_dir / bench_name
        bench_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if we have cached probe times - if so, use them and skip probing
        probe_cache_file = out_dir / f"{bench_name}_probe_times.txt"
        units = {}
        probe_times = {}
        
        if probe_cache_file.exists():
            # Load cached probe times and extract units from it
            # Format: unit_name elapsed_time perfspect_command
            # Note: unit_name can have spaces, so split from the right
            with open(probe_cache_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Split by "perfspect" to separate unit+duration from command
                    if ' perfspect ' not in line:
                        continue
                    
                    unit_and_duration, cmd_part = line.split(' perfspect ', 1)
                    # The last token in unit_and_duration is the elapsed time
                    tokens = unit_and_duration.rsplit(maxsplit=1)
                    if len(tokens) != 2:
                        continue
                    
                    unit_name = tokens[0]
                    try:
                        elapsed = float(tokens[1])
                        cmd = f"perfspect {cmd_part}"
                        units[unit_name] = 1e6  # Default throughput estimate
                        probe_times[unit_name] = (elapsed, cmd)
                    except ValueError:
                        continue
            
            print(f"  Loaded {len(units)} cached units from probe times")
            
            # Count existing summary files in bench_dir
            existing_summaries = list(bench_dir.glob("*_summary.csv"))
            
            if len(existing_summaries) == len(units):
                print(f"  Found all {len(existing_summaries)} summary files - skipping perfspect")
                continue
            else:
                print(f"  Found {len(existing_summaries)} summary files but have {len(units)} units - will re-run")
        else:
            # No cache - probe normally
            units, probe_times = discovery.run_benchmark_get_units(bench_path)
        
        if not units:
            print(f"  no units found for {bench_name}")
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
                # Wrap pattern in double quotes for shell execution
                bench_cmd = [str(bench_path), '--bm_regex', f'"{regex_pattern}"']

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
