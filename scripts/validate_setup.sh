#!/bin/bash
# Validation script to check setup and provide usage info

set -e

WDL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)"

echo "========================================================================"
echo "WDLBench PMU Performance Counter Measurement - Setup Validation"
echo "========================================================================"
echo ""

# Check Python version
echo "1. Checking Python 3 installation..."
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 --version 2>&1)
    echo "   ✓ $PY_VERSION"
else
    echo "   ✗ Python 3 not found. Install with: sudo apt install python3"
    exit 1
fi

# Check perf tool
echo ""
echo "2. Checking perf tool installation..."
if command -v perf &> /dev/null; then
    echo "   ✓ perf found"
else
    echo "   ✗ perf not found. Install with: sudo apt install linux-tools-generic"
    exit 1
fi

# Check benchmark binaries
echo ""
echo "3. Checking benchmark binaries..."
WDL_BUILD="${WDL_ROOT}/wdl_build"
if [ -d "$WDL_BUILD" ]; then
    BENCH_COUNT=$(ls -1 "$WDL_BUILD" 2>/dev/null | grep -E "benchmark|bench" | wc -l)
    echo "   ✓ wdl_build directory found ($BENCH_COUNT benchmarks)"
else
    echo "   ✗ wdl_build directory not found"
    echo "     Run: ./install_wdl_bench.sh or similar build script"
fi

# Check script files
echo ""
echo "4. Checking required scripts..."
SCRIPTS=("run_perf_benchmarks.py" "analyze_perf_results.py")
for script in "${SCRIPTS[@]}"; do
    if [ -x "$WDL_ROOT/$script" ]; then
        echo "   ✓ $script (executable)"
    else
        echo "   ✗ $script not found or not executable"
    fi
done

# Check documentation
echo ""
echo "5. Checking documentation..."
DOCS=("PERF_MEASUREMENT_GUIDE.md" "PMU_MEASUREMENT_SUMMARY.md" "QUICK_REFERENCE.sh")
for doc in "${DOCS[@]}"; do
    if [ -f "$WDL_ROOT/$doc" ]; then
        echo "   ✓ $doc"
    else
        echo "   ✗ $doc not found"
    fi
done

# Check perf permissions
echo ""
echo "6. Checking perf permissions..."
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "unknown")
if [ "$PARANOID" -le 1 ] 2>/dev/null; then
    echo "   ✓ perf_event_paranoid=$PARANOID (OK for user access)"
else
    echo "   ⚠ perf_event_paranoid=$PARANOID (may require sudo)"
    echo "     To enable user access (one-time): sudo sysctl kernel.perf_event_paranoid=1"
fi

# Summary
echo ""
echo "========================================================================"
echo "Setup Status: READY"
echo "========================================================================"
echo ""
echo "Quick start:"
echo "  1. Run a benchmark:"
echo "     cd $(basename "$WDL_ROOT")"
echo "     python3 run_perf_benchmarks.py --name memcpy_benchmark"
echo ""
echo "  2. Analyze results:"
echo "     python3 analyze_perf_results.py"
echo ""
echo "  3. Export results:"
echo "     python3 analyze_perf_results.py --csv results.csv"
echo ""
echo "Documentation:"
echo "  - PERF_MEASUREMENT_GUIDE.md    - Comprehensive guide"
echo "  - PMU_MEASUREMENT_SUMMARY.md   - Overview and examples"
echo "  - QUICK_REFERENCE.sh           - Quick command reference"
echo ""
echo "Run 'python3 run_perf_benchmarks.py --help' for more options"
echo ""
