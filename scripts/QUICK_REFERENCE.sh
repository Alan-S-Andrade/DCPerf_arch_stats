#!/bin/bash
# Quick reference for running microbenchmarks with PMU measurements

# ==============================================================================
# QUICK START EXAMPLES
# ==============================================================================

# 1. Run memcpy_benchmark with PMU measurements
python3 run_perf_benchmarks.py --name memcpy_benchmark

# 2. Run all benchmarks (takes longer)
python3 run_perf_benchmarks.py

# 3. Analyze results
python3 analyze_perf_results.py

# 4. Export results to CSV
python3 analyze_perf_results.py --csv results.csv

# 5. Compare two benchmarks
python3 analyze_perf_results.py --compare memcpy_benchmark memset_benchmark

# ==============================================================================
# MANUAL PERF COMMANDS (For single units)
# ==============================================================================

cd wdl_build

# Run memcpy COLD variants with perf
sudo perf stat -a \
  -e cycles,instructions,L1-icache-load-misses,iTLB-loads,iTLB-load-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,LLC-load-misses,branch-load-misses,branch-misses,r2424 \
  -o memcpy_cold_perf.json --json \
  ./memcpy_benchmark --bm_pattern='.*COLD.*'

# Run memcpy HOT variants
sudo perf stat -a \
  -e cycles,instructions,L1-icache-load-misses,iTLB-loads,iTLB-load-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,LLC-load-misses,branch-load-misses,branch-misses,r2424 \
  -o memcpy_hot_perf.json --json \
  ./memcpy_benchmark --bm_pattern='.*HOT.*'

# Run avx2 memory operations
sudo perf stat -a \
  -e cycles,instructions,L1-icache-load-misses,iTLB-loads,iTLB-load-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,LLC-load-misses,branch-load-misses,branch-misses,r2424 \
  -o memcpy_avx2_perf.json --json \
  ./memcpy_benchmark --bm_pattern='avx2_memops.*'

# ==============================================================================
# VIEWING RESULTS
# ==============================================================================

# Summary of all measurements
cat ../perf_results/*_perf.json | grep -i "cycles\|instructions"

# Count number of units measured
ls ../perf_results/*_perf.json | wc -l

# List all benchmark units measured
ls ../perf_results/*_perf.txt | sed 's/.*\///;s/_perf.txt//'

# ==============================================================================
# AVAILABLE BENCHMARK PATTERNS
# ==============================================================================

# memcpy_benchmark patterns:
#   ./memcpy_benchmark --bm_pattern='.*COLD.*'        # Cold cache variants
#   ./memcpy_benchmark --bm_pattern='.*HOT.*'         # Hot cache variants
#   ./memcpy_benchmark --bm_pattern='.*folly'         # Folly implementation only
#   ./memcpy_benchmark --bm_pattern='.*glibc'         # Glibc implementation only
#   ./memcpy_benchmark --bm_pattern='avx2_memops.*'   # AVX2 only
#   ./memcpy_benchmark --bm_pattern='scalar_memops.*' # Scalar only
#   ./memcpy_benchmark --bm_pattern='bench\(.*1024.*\)' # Size 1024 variants

# Generic pattern examples:
#   ./benchmark --bm_pattern='specific_test_name'
#   ./benchmark --bm_pattern='.*substring.*'
#   ./benchmark --bm_pattern='^StartsWith'

# ==============================================================================
# CUSTOMIZING PMU EVENTS
# ==============================================================================

# Change PMU events:
python3 run_perf_benchmarks.py --name memcpy_benchmark \
  --events "cycles,instructions,cache-references,cache-misses"

# Available events (check with: perf list):
#   cycles
#   instructions
#   cache-references
#   cache-misses
#   branch-instructions
#   branch-misses
#   stalled-cycles-frontend
#   stalled-cycles-backend
#   LLC-loads
#   LLC-load-misses
#   L1-dcache-loads
#   L1-dcache-load-misses
#   L1-icache-load-misses
#   dTLB-loads
#   dTLB-load-misses
#   iTLB-loads
#   iTLB-load-misses

# ==============================================================================
# TROUBLESHOOTING
# ==============================================================================

# Give user permission to run perf (requires sudo once):
# sudo sysctl kernel.perf_event_paranoid=0

# Check if benchmarks are built:
# ls -la wdl_build/memcpy_benchmark

# Show sample benchmark output:
# ./wdl_build/memcpy_benchmark --json | head -100

# Check available perf events on your system:
# perf list | less

# ==============================================================================
