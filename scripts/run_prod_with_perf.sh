#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Enhanced version to run individual microbenchmark units with perf stat PMU measurements

set -Eeo pipefail

BREPS_LFILE=/tmp/wdl_log.txt

function benchreps_tell_state () {
    date +"%Y-%m-%d_%T ${1}" >> $BREPS_LFILE
}

# Constants
WDL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)
WDL_DATASETS="${WDL_ROOT}/datasets"
WDL_BUILD="${WDL_ROOT}/wdl_build"
PERF_RESULTS_DIR="${WDL_ROOT}/perf_results"

# PMU Events to measure
PERF_EVENTS="cycles,instructions,L1-icache-load-misses,iTLB-loads,iTLB-load-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,LLC-load-misses,branch-load-misses,branch-misses,r2424"

show_help() {
cat <<EOF
Usage: ${0##*/} [-h] [--name benchmark_name] [--duration seconds]

    -h                  Display this help and exit
    --name              Specific benchmark to run (e.g., memcpy_benchmark)
    --duration          Duration in seconds for each benchmark unit (default: 30)
EOF
}

# Benchmarks with JSON output (can extract individual units)
json_benchmark_list="memcpy_benchmark memset_benchmark bench-memcmp hash_hash_benchmark hash_checksum_benchmark synchronization_lifo_sem_bench synchronization_small_locks_benchmark container_hash_maps_bench ProtocolBench VarintUtilsBench random_benchmark benchsleef128 benchsleef256 benchsleef512"

# Non-JSON benchmarks that need custom handling
non_json_benchmarks="openssl libaegis_benchmark lzbench vdso_bench xxhash_benchmark concurrency_concurrent_hash_map_bench erasure_code_perf"

declare -A prod_benchmark_config=(
    ['random_benchmark']="--bm_regex=xoshiro --json"
    ['memcpy_benchmark']="--json"
    ['memset_benchmark']="--json"
    ['hash_hash_benchmark']="--bm_regex=RapidHash --json"
    ['hash_checksum_benchmark']="--json"
    ['synchronization_lifo_sem_bench']="--bm_min_iters=1000000 --json"
    ['synchronization_small_locks_benchmark']="--bm_min_iters=1000000 --bm_regex=folly_RWSpinlock --json"
    ['container_hash_maps_bench']="--bm_regex=\"f14(vec)|(val)\" --json"
    ['ProtocolBench']="--bm_regex=\"(^Binary)|(^Compact)Protocol\" --json"
    ['VarintUtilsBench']="--json"
    ['bench-memcmp']=""
    ['benchsleef128']="--benchmark_format=json"
    ['benchsleef256']="--benchmark_format=json"
    ['benchsleef512']="--benchmark_format=json"
)

# Function to extract benchmark unit names from JSON output
extract_benchmark_units() {
    local json_file="$1"
    local benchmark_name="$2"
    
    if [ ! -f "$json_file" ]; then
        echo "JSON file not found: $json_file" >&2
        return 1
    fi
    
    # Extract benchmark names from JSON (handles different JSON formats)
    python3 << 'PYTHON_EOF'
import json
import sys
try:
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)
    
    # Handle different JSON formats
    if isinstance(data, dict):
        if 'benchmarks' in data:
            # Google Benchmark format
            for bench in data['benchmarks']:
                if 'name' in bench:
                    print(bench['name'])
        else:
            # Direct key-value format
            for key in data.keys():
                if not key.startswith('%'):
                    print(key)
    elif isinstance(data, list):
        # Array of benchmarks
        for item in data:
            if isinstance(item, dict) and 'name' in item:
                print(item['name'])
except json.JSONDecodeError:
    # Try to parse as folly JSON format (one benchmark per line with name prefix)
    with open(sys.argv[1], 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('{'):
                # Extract benchmark name from folly format
                if ':' in line:
                    print(line.split(':')[0])
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_EOF
}

# Function to run a single benchmark unit with perf stat
run_benchmark_with_perf() {
    local benchmark_name="$1"
    local unit_name="$2"
    local benchmark_binary="$3"
    local config="${4:-}"
    
    local output_file="${PERF_RESULTS_DIR}/${benchmark_name}_${unit_name// /_}_perf.txt"
    local json_output_file="${PERF_RESULTS_DIR}/${benchmark_name}_${unit_name// /_}_perf.json"
    
    # Sanitize output filename
    output_file="${output_file//\//_}"
    json_output_file="${json_output_file//\//_}"
    
    echo "[$(date)] Running: $benchmark_name::$unit_name"
    
    # Build perf command with pattern matching for the specific unit
    local perf_cmd="sudo perf stat -a -e $PERF_EVENTS -o $json_output_file --json"
    
    # Add bm_pattern filter if configuration doesn't already have it
    local benchmark_cmd="${benchmark_binary} ${config}"
    if [[ ! "$benchmark_cmd" =~ "--bm_pattern" ]] && [[ ! "$benchmark_cmd" =~ "--bm_regex" ]]; then
        # For benchmarks with unit names, add pattern matching
        # Escape special regex characters in unit name
        local escaped_unit=$(echo "$unit_name" | sed 's/[[\.*^$/]/\\&/g')
        benchmark_cmd="${benchmark_binary} --bm_pattern='$escaped_unit' ${config}"
    fi
    
    # Execute perf stat wrapping the benchmark
    local cmd="$perf_cmd -- $benchmark_cmd 2>&1 | tee -a $output_file"
    echo "Command: $cmd"
    eval "$cmd" || true
    
    # Convert perf JSON output if available
    if [ -f "$json_output_file" ]; then
        echo "Perf results saved to: $json_output_file"
    fi
}

# Function to run benchmark and extract all units
run_benchmark_with_all_units() {
    local benchmark="$1"
    local out_file="${WDL_ROOT}/out_${benchmark}.json"
    
    echo ""
    echo "=========================================="
    echo "Processing benchmark: $benchmark"
    echo "=========================================="
    
    # First, run the benchmark to generate JSON with all units
    pushd "${WDL_ROOT}"
    
    local config="${prod_benchmark_config[$benchmark]:-}"
    
    # Run benchmark and capture output
    bash -c "./${benchmark} ${config}" > "$out_file" 2>&1 || true
    
    # Extract and run each unit with perf
    if [ -f "$out_file" ]; then
        # Try to extract unit names and run each with perf
        mapfile -t units < <(extract_benchmark_units "$out_file" "$benchmark")
        
        if [ ${#units[@]} -eq 0 ]; then
            echo "Warning: No units extracted from $out_file, running benchmark once with perf"
            run_benchmark_with_perf "$benchmark" "all" "./${benchmark}" "$config"
        else
            echo "Found ${#units[@]} benchmark units for $benchmark"
            for unit in "${units[@]}"; do
                if [ -n "$unit" ]; then
                    run_benchmark_with_perf "$benchmark" "$unit" "./${benchmark}" "$config"
                fi
            done
        fi
    else
        echo "Warning: Could not generate output for $benchmark"
    fi
    
    popd
}

main() {
    local benchmark_to_run="none"
    local perf_duration=30
    
    while :; do
        case $1 in
            --name)
                benchmark_to_run="$2"
                shift 2
                ;;
            --duration)
                perf_duration="$2"
                shift 2
                ;;
            -h|--help)
                show_help >&2
                exit 1
                ;;
            *)
                break
                ;;
        esac
    done
    
    # Create results directory
    mkdir -p "$PERF_RESULTS_DIR"
    
    benchreps_tell_state "starting perf measurement"
    
    if [ "$benchmark_to_run" != "none" ]; then
        run_benchmark_with_all_units "$benchmark_to_run"
    else
        # Run all JSON-based benchmarks
        for benchmark in $json_benchmark_list; do
            if [ -f "${WDL_BUILD}/${benchmark}" ]; then
                run_benchmark_with_all_units "$benchmark"
            fi
        done
    fi
    
    benchreps_tell_state "perf measurement complete"
    
    echo ""
    echo "=========================================="
    echo "Perf Results Summary"
    echo "=========================================="
    echo "Results saved to: $PERF_RESULTS_DIR"
    ls -lh "$PERF_RESULTS_DIR" | tail -20
}

main "$@"
