#!/bin/bash
# Quick Start Guide - Full PMU Measurement of Folly Benchmarks

# ==============================================================================
# WHAT YOU'LL GET
# ==============================================================================
# For EVERY microbenchmark unit (bench(0_to_1024_COLD_folly), etc.):
#   - Exact cycle count
#   - Instruction count  
#   - Cache miss counts
#   - TLB statistics
#   - Branch prediction misses
#   - All other configured PMU events
#
# Measured for the ENTIRE execution duration of each unit.

# ==============================================================================
# QUICKEST START - MEMCPY ONLY
# ==============================================================================

cd /users/alanuiuc/DCPerf/packages/wdl_bench

# Run memcpy_benchmark with full PMU measurement
# This measures units like:
#   - bench(0_to_1024_COLD_folly)
#   - bench(0_to_1024_HOT_folly)
#   - bench(0_to_32768_COLD_folly)
#   - scalar_memops(aligned_reads)
#   - avx2_memops(aligned_reads)
#   - ... and ~50 more units

python3 run_detailed_perf_benchmarks.py --benchmark memcpy

# Results saved to: perf_results_detailed/
# Example files:
#   perf_results_detailed/memcpy_benchmark_bench(0_to_1024_COLD_folly)_perf.json
#   perf_results_detailed/memcpy_benchmark_bench(0_to_1024_COLD_folly)_perf.txt

# ==============================================================================
# ANALYZE RESULTS
# ==============================================================================

# Print summary
python3 analyze_perf_results.py --results-dir perf_results_detailed

# Export to CSV for spreadsheet analysis
python3 analyze_perf_results.py --results-dir perf_results_detailed \
  --csv memcpy_results.csv

# Compare with memset
python3 analyze_perf_results.py --results-dir perf_results_detailed \
  --compare memcpy_benchmark memset_benchmark

# ==============================================================================
# RUN ALL BENCHMARKS (takes longer, but comprehensive)
# ==============================================================================

python3 run_detailed_perf_benchmarks.py

# This runs ALL folly benchmarks:
# - memcpy (~60 units)
# - memset (~40 units)  
# - hash (~15 units)
# - synchronization (~20 units)
# - serialization/protocol (~8 units)
# - random, checksum, etc. (~50 more units)
# Total: ~200+ individual test units measured with full PMU data

# ==============================================================================
# UNDERSTAND THE OUTPUT
# ==============================================================================

# Each unit gets TWO files:

# 1. JSON file with perf data
cat perf_results_detailed/memcpy_benchmark_bench\(0_to_1024_COLD_folly\)_perf.json

# Output:
# {
#   "data": [
#     {"event": "cycles", "value": 234567, "unit": "c"},
#     {"event": "instructions", "value": 123456, "unit": "c"},
#     {"event": "L1-dcache-load-misses", "value": 45, "unit": "c"},
#     ...
#   ]
# }

# 2. Text file with human-readable info
cat perf_results_detailed/memcpy_benchmark_bench\(0_to_1024_COLD_folly\)_perf.txt

# Shows:
# - Benchmark name and unit name
# - Estimated throughput
# - Actual measurement duration
# - Perf stat output

# ==============================================================================
# DURATION CALCULATION (HOW IT WORKS)
# ==============================================================================

# For each unit, the script:
# 1. Extracts benchmark name (e.g., "bench(0_to_1024_COLD_folly)")
# 2. Extracts throughput from JSON (e.g., 11.20M iters/s)
# 3. Calculates duration to capture ~50,000 iterations:
#
#    Duration = 50,000 iters / throughput
#    Example: 50,000 / 11,200,000 = 4.46ms
#
# 4. Runs perf stat for that duration
# 5. Captures all PMU events during that time

# Examples:
# - Very fast unit (1.11G iters/s): 50ms measurement (minimum)
# - Fast unit (450M iters/s): 110us measurement
# - Medium unit (11M iters/s): 4.5ms measurement
# - Slow unit (2.5M iters/s): 20ms measurement

# ==============================================================================
# EXAMPLE: MEASURE SPECIFIC UNITS ONLY
# ==============================================================================

# Measure only memcpy with custom PMU events
python3 run_detailed_perf_benchmarks.py \
  --benchmark memcpy \
  --events "cycles,instructions,cache-misses,branch-misses"

# Measure hash benchmarks
python3 run_detailed_perf_benchmarks.py \
  --benchmark hash \
  --output-dir ./hash_perf_results

# Measure synchronization tests
python3 run_detailed_perf_benchmarks.py \
  --benchmark synchronization

# ==============================================================================
# VIEW/ANALYZE RESULTS
# ==============================================================================

# Quick summary
python3 analyze_perf_results.py --results-dir perf_results_detailed

# Top 20 events by value
python3 analyze_perf_results.py --results-dir perf_results_detailed --top-events 20

# Export everything to Excel with per-benchmark sheets
python3 analyze_perf_results.py --results-dir perf_results_detailed \
  --excel analysis.xlsx

# ==============================================================================
# WHAT EACH PMU EVENT MEANS
# ==============================================================================

cat <<'EVENTS'

cycles                      = CPU clock cycles (performance = low cycles)
instructions                = Instructions executed (efficiency = high IPC)
L1-icache-load-misses       = Instruction cache misses (code locality issue)
iTLB-loads                  = Instruction TLB lookups
iTLB-load-misses            = Instruction TLB misses (code page walks)
L1-dcache-loads             = Data cache loads
L1-dcache-load-misses       = Data cache load misses (working set too large)
L1-dcache-stores            = Data cache stores
LLC-load-misses             = Last-level cache misses (main memory access)
branch-load-misses          = Branch prediction misses
branch-misses               = Branch mispredictions

Key metrics:
  IPC = instructions / cycles  (> 4 is good)
  L1 miss rate = L1-dcache-load-misses / L1-dcache-loads
  Cycles per instruction = cycles / instructions

EVENTS

# ==============================================================================
# EXAMPLE WORKFLOW
# ==============================================================================

# 1. Measure memcpy in detail
echo "Measuring memcpy benchmark units..."
python3 run_detailed_perf_benchmarks.py --benchmark memcpy

# 2. Check how many units were measured
echo "Units measured:"
ls perf_results_detailed/memcpy_benchmark_*.json | wc -l

# 3. List all units
echo "All units tested:"
ls perf_results_detailed/memcpy_benchmark_*_perf.txt | \
  xargs -I {} basename {} _perf.txt | \
  sed 's/memcpy_benchmark_//'

# 4. View results for specific unit
echo "Results for 0_to_1024_COLD variant:"
cat perf_results_detailed/memcpy_benchmark_bench\(0_to_1024_COLD_folly\)_perf.txt

# 5. Compare COLD vs HOT
echo "Cycles for COLD variants:"
grep '"cycles"' perf_results_detailed/memcpy_benchmark_*COLD*_perf.json

echo "Cycles for HOT variants:"
grep '"cycles"' perf_results_detailed/memcpy_benchmark_*HOT*_perf.json

# ==============================================================================
# TROUBLESHOOTING
# ==============================================================================

# If you get "PERMISSION DENIED":
sudo python3 run_detailed_perf_benchmarks.py --benchmark memcpy

# If benchmarks not found:
ls /users/alanuiuc/DCPerf/benchmarks/wdl_bench/wdl_build/
# If empty, build them first:
cd /users/alanuiuc/DCPerf/packages/wdl_bench
./install_wdl_bench.sh

# If perf events not available:
perf list | head -20
# Use only available events:
python3 run_detailed_perf_benchmarks.py --benchmark memcpy \
  --events "cycles,instructions,cache-references,cache-misses"

# ==============================================================================
# DETAILED DOCUMENTATION
# ==============================================================================

# For complete guide, see:
cat DETAILED_PERF_MEASUREMENT.md

# For original quick reference:
cat QUICK_REFERENCE.sh

# ==============================================================================
