#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Script to analyze perf stat results

import json
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import re

class PerfResultsAnalyzer:
    """Analyzes perf stat results from benchmark runs."""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.all_results = {}
        self.load_results()
    
    def load_results(self):
        """Load all perf result JSON files."""
        
        if not self.results_dir.exists():
            print(f"Error: Results directory not found: {self.results_dir}", file=sys.stderr)
            return
        
        json_files = list(self.results_dir.glob("*_perf.json"))
        
        if not json_files:
            print(f"No perf result JSON files found in {self.results_dir}", file=sys.stderr)
            return
        
        print(f"Loading {len(json_files)} perf result files...")
        
        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract benchmark and unit name from filename
                filename = json_file.stem  # Remove .json
                parts = filename.rsplit('_perf', 1)[0].split('_', 1)
                
                if len(parts) == 2:
                    bench_name, unit_name = parts
                    unit_name = unit_name.replace('_', ' ')  # Restore spaces
                else:
                    bench_name = parts[0]
                    unit_name = "all"
                
                key = f"{bench_name}::{unit_name}"
                self.all_results[key] = self._parse_perf_json(data)
            
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse {json_file}: {e}", file=sys.stderr)
            except Exception as e:
                print(f"Error processing {json_file}: {e}", file=sys.stderr)
    
    def _parse_perf_json(self, data: dict) -> Dict:
        """Extract relevant metrics from perf stat JSON."""
        
        metrics = {}
        
        # perf stat JSON format has 'data' key with list of metrics
        if 'data' in data:
            for item in data['data']:
                if isinstance(item, dict):
                    event = item.get('event', '')
                    value = item.get('value', 0)
                    unit = item.get('unit', '')
                    
                    if event:
                        # Store with unit
                        metrics[event] = {
                            'value': float(value) if isinstance(value, (int, float)) else 0,
                            'unit': unit
                        }
        
        return metrics
    
    def create_summary_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with all results for easy analysis."""
        
        rows = []
        
        for result_key, metrics in self.all_results.items():
            row = {'benchmark_unit': result_key}
            
            # Add all metrics
            for event_name, event_data in metrics.items():
                row[event_name] = event_data['value']
            
            rows.append(row)
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        return df
    
    def print_summary(self):
        """Print a summary of all results."""
        
        if not self.all_results:
            print("No results to summarize")
            return
        
        print(f"\n{'='*80}")
        print("PERF RESULTS SUMMARY")
        print(f"{'='*80}\n")
        
        print(f"Total benchmark units measured: {len(self.all_results)}\n")
        
        # Group by benchmark
        benchmarks = {}
        for result_key in self.all_results.keys():
            bench_name = result_key.split('::')[0]
            if bench_name not in benchmarks:
                benchmarks[bench_name] = []
            benchmarks[bench_name].append(result_key)
        
        for bench_name in sorted(benchmarks.keys()):
            units = benchmarks[bench_name]
            print(f"{bench_name}: {len(units)} units")
    
    def export_csv(self, output_file: str = "perf_results.csv"):
        """Export results to CSV."""
        
        df = self.create_summary_dataframe()
        
        if df.empty:
            print("No data to export", file=sys.stderr)
            return
        
        output_path = Path(output_file)
        df.to_csv(output_path, index=False)
        print(f"Results exported to: {output_path}")
        print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    def export_excel(self, output_file: str = "perf_results.xlsx"):
        """Export results to Excel with multiple sheets."""
        
        try:
            import openpyxl
        except ImportError:
            print("openpyxl not installed. Install with: pip install openpyxl", file=sys.stderr)
            return
        
        df = self.create_summary_dataframe()
        
        if df.empty:
            print("No data to export", file=sys.stderr)
            return
        
        output_path = Path(output_file)
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Full data
            df.to_excel(writer, sheet_name='All Results', index=False)
            
            # Separate sheets for each benchmark
            benchmarks = {}
            for _, row in df.iterrows():
                bench_name = row['benchmark_unit'].split('::')[0]
                if bench_name not in benchmarks:
                    benchmarks[bench_name] = []
                benchmarks[bench_name].append(row)
            
            for bench_name, rows in benchmarks.items():
                bench_df = pd.DataFrame(rows)
                sheet_name = bench_name[:31]  # Excel sheet name limit
                bench_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"Results exported to: {output_path}")
        print(f"Total sheets: {len(benchmarks) + 1}")
    
    def print_top_events(self, top_n: int = 10):
        """Print top events by value across all benchmarks."""
        
        df = self.create_summary_dataframe()
        
        if df.empty:
            print("No data available")
            return
        
        # Get numeric columns (excluding benchmark_unit)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        print(f"\n{'='*80}")
        print(f"TOP {top_n} EVENTS BY MEDIAN VALUE")
        print(f"{'='*80}\n")
        
        for col in numeric_cols:
            median_val = df[col].median()
            max_val = df[col].max()
            min_val = df[col].min()
            
            print(f"{col:40s}: median={median_val:15.0f} max={max_val:15.0f} min={min_val:15.0f}")
    
    def compare_benchmarks(self, bench1: str, bench2: str):
        """Compare two benchmarks."""
        
        bench1_results = {k: v for k, v in self.all_results.items() if k.startswith(f"{bench1}::")}
        bench2_results = {k: v for k, v in self.all_results.items() if k.startswith(f"{bench2}::")}
        
        if not bench1_results or not bench2_results:
            print(f"One or both benchmarks not found", file=sys.stderr)
            return
        
        print(f"\n{'='*80}")
        print(f"COMPARISON: {bench1} vs {bench2}")
        print(f"{'='*80}\n")
        
        # Get common events
        events1 = set()
        events2 = set()
        
        for metrics in bench1_results.values():
            events1.update(metrics.keys())
        
        for metrics in bench2_results.values():
            events2.update(metrics.keys())
        
        common_events = events1 & events2
        
        if not common_events:
            print("No common events found")
            return
        
        for event in sorted(common_events):
            vals1 = [m[event]['value'] for m in bench1_results.values() if event in m]
            vals2 = [m[event]['value'] for m in bench2_results.values() if event in m]
            
            if vals1 and vals2:
                avg1 = sum(vals1) / len(vals1)
                avg2 = sum(vals2) / len(vals2)
                diff_pct = ((avg2 - avg1) / avg1 * 100) if avg1 > 0 else 0
                
                print(f"{event:40s}: {avg1:15.0f} vs {avg2:15.0f} ({diff_pct:+7.1f}%)")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze perf stat results")
    parser.add_argument(
        '--results-dir',
        default='./perf_results',
        help='Path to perf results directory'
    )
    parser.add_argument(
        '--csv',
        help='Export results to CSV file'
    )
    parser.add_argument(
        '--excel',
        help='Export results to Excel file'
    )
    parser.add_argument(
        '--compare',
        nargs=2,
        metavar=('BENCH1', 'BENCH2'),
        help='Compare two benchmarks'
    )
    parser.add_argument(
        '--top-events',
        type=int,
        default=10,
        help='Show top N events'
    )
    
    args = parser.parse_args()
    
    # Change to script directory
    script_dir = Path(__file__).parent.absolute()
    if Path(args.results_dir).is_relative_to(script_dir) or not Path(args.results_dir).is_absolute():
        results_dir = script_dir / args.results_dir
    else:
        results_dir = Path(args.results_dir)
    
    analyzer = PerfResultsAnalyzer(str(results_dir))
    
    if args.compare:
        analyzer.compare_benchmarks(args.compare[0], args.compare[1])
    else:
        analyzer.print_summary()
        analyzer.print_top_events(args.top_events)
    
    if args.csv:
        analyzer.export_csv(args.csv)
    
    if args.excel:
        analyzer.export_excel(args.excel)

if __name__ == '__main__':
    main()
