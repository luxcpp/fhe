#!/bin/bash
# FHE Backend Comparison Benchmark
# Compares CPU, MLX (Metal GPU), and WebGPU backends
#
# Usage: ./run_backend_comparison.sh [--cpu] [--mlx] [--webgpu] [--all]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FHE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_CPU="$FHE_ROOT/build-local"
BUILD_MLX="$FHE_ROOT/build_mlx"
BUILD_WEBGPU="$FHE_ROOT/build-webgpu"

# Benchmark parameters
BENCH_TIME="${BENCH_TIME:-5}"  # seconds per benchmark
BENCH_ITERATIONS="${BENCH_ITERATIONS:-3}"

# Output
RESULTS_DIR="$SCRIPT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$RESULTS_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check available backends
check_backends() {
    echo ""
    echo "=============================================="
    echo "         FHE Backend Availability Check       "
    echo "=============================================="

    local cpu_ok=0
    local mlx_ok=0
    local webgpu_ok=0

    # CPU backend (always available if built)
    if [[ -d "$BUILD_CPU" ]] && [[ -f "$BUILD_CPU/bin/lib-benchmark" ]]; then
        log_success "CPU backend: AVAILABLE"
        cpu_ok=1
    else
        log_warn "CPU backend: NOT BUILT (run: cmake -B build-local && make -C build-local)"
    fi

    # MLX (Metal) backend
    if [[ -d "$BUILD_MLX" ]] && [[ -f "$BUILD_MLX/lib/libOPENFHEcore.dylib" ]]; then
        log_success "MLX (Metal) backend: AVAILABLE"
        mlx_ok=1
    else
        log_warn "MLX backend: NOT BUILT (run: cmake -B build_mlx -DWITH_MLX=ON && make -C build_mlx)"
    fi

    # WebGPU backend
    if [[ -d "$BUILD_WEBGPU" ]] && [[ -f "$BUILD_WEBGPU/bin/lib-benchmark" ]]; then
        log_success "WebGPU backend: AVAILABLE"
        webgpu_ok=1
    else
        log_warn "WebGPU backend: NOT BUILT (run: cmake -B build-webgpu -DWITH_WEBGPU=ON && make -C build-webgpu)"
    fi

    echo ""

    export CPU_AVAILABLE=$cpu_ok
    export MLX_AVAILABLE=$mlx_ok
    export WEBGPU_AVAILABLE=$webgpu_ok
}

# Run benchmark with specific backend
run_benchmark() {
    local backend=$1
    local build_dir=$2
    local output_file=$3

    log_info "Running $backend benchmarks..."

    if [[ ! -f "$build_dir/bin/lib-benchmark" ]]; then
        log_error "$backend benchmark binary not found"
        return 1
    fi

    # Set library path
    export DYLD_LIBRARY_PATH="$build_dir/lib:$DYLD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="$build_dir/lib:$LD_LIBRARY_PATH"

    # Run Google Benchmark with JSON output
    "$build_dir/bin/lib-benchmark" \
        --benchmark_format=json \
        --benchmark_out="$output_file" \
        --benchmark_repetitions="$BENCH_ITERATIONS" \
        --benchmark_min_time="${BENCH_TIME}s" \
        2>&1 | tee "${output_file%.json}.log"

    log_success "$backend benchmarks completed: $output_file"
}

# Run TFHE gate benchmarks
run_tfhe_benchmark() {
    local backend=$1
    local build_dir=$2
    local output_file=$3

    log_info "Running $backend TFHE gate benchmarks..."

    if [[ ! -f "$build_dir/bin/binfhe-ginx" ]]; then
        log_warn "$backend TFHE benchmark not available"
        return 0
    fi

    export DYLD_LIBRARY_PATH="$build_dir/lib:$DYLD_LIBRARY_PATH"

    "$build_dir/bin/binfhe-ginx" \
        --benchmark_format=json \
        --benchmark_out="$output_file" \
        --benchmark_repetitions="$BENCH_ITERATIONS" \
        2>&1 | tee "${output_file%.json}.log"

    log_success "$backend TFHE benchmarks completed"
}

# Run CKKS benchmarks
run_ckks_benchmark() {
    local backend=$1
    local build_dir=$2
    local output_file=$3

    log_info "Running $backend CKKS benchmarks..."

    # Try different benchmark binaries
    local bench_bin=""
    for candidate in "ckks_bench" "serialize-ckks" "lib-benchmark"; do
        if [[ -f "$build_dir/bin/$candidate" ]]; then
            bench_bin="$build_dir/bin/$candidate"
            break
        fi
    done

    if [[ -z "$bench_bin" ]]; then
        log_warn "$backend CKKS benchmark not available"
        return 0
    fi

    export DYLD_LIBRARY_PATH="$build_dir/lib:$DYLD_LIBRARY_PATH"

    "$bench_bin" \
        --benchmark_format=json \
        --benchmark_out="$output_file" \
        --benchmark_filter="CKKS" \
        --benchmark_repetitions="$BENCH_ITERATIONS" \
        2>&1 | tee "${output_file%.json}.log"

    log_success "$backend CKKS benchmarks completed"
}

# Parse JSON results and create comparison table
generate_comparison() {
    local output_file="$RESULTS_DIR/comparison_${TIMESTAMP}.md"

    log_info "Generating comparison report..."

    cat > "$output_file" << 'HEADER'
# FHE Backend Comparison Results

## Test Configuration
- Platform: Apple Silicon (M1 Max)
- Date: $(date)
- Benchmark iterations: ${BENCH_ITERATIONS}
- Min time per benchmark: ${BENCH_TIME}s

## Results Summary

| Operation | CPU (ms) | MLX/Metal (ms) | WebGPU (ms) | MLX Speedup | WebGPU Speedup |
|-----------|----------|----------------|-------------|-------------|----------------|
HEADER

    # Parse results (simplified - would use jq for real parsing)
    if command -v jq &> /dev/null; then
        for op in "NTT" "INTT" "KeyGen" "Encrypt" "Decrypt" "Add" "Mult" "Bootstrap"; do
            local cpu_time="N/A"
            local mlx_time="N/A"
            local webgpu_time="N/A"

            if [[ -f "$RESULTS_DIR/cpu_${TIMESTAMP}.json" ]]; then
                cpu_time=$(jq -r ".benchmarks[] | select(.name | contains(\"$op\")) | .real_time" \
                    "$RESULTS_DIR/cpu_${TIMESTAMP}.json" 2>/dev/null | head -1 || echo "N/A")
            fi

            if [[ -f "$RESULTS_DIR/mlx_${TIMESTAMP}.json" ]]; then
                mlx_time=$(jq -r ".benchmarks[] | select(.name | contains(\"$op\")) | .real_time" \
                    "$RESULTS_DIR/mlx_${TIMESTAMP}.json" 2>/dev/null | head -1 || echo "N/A")
            fi

            if [[ -f "$RESULTS_DIR/webgpu_${TIMESTAMP}.json" ]]; then
                webgpu_time=$(jq -r ".benchmarks[] | select(.name | contains(\"$op\")) | .real_time" \
                    "$RESULTS_DIR/webgpu_${TIMESTAMP}.json" 2>/dev/null | head -1 || echo "N/A")
            fi

            # Calculate speedups
            local mlx_speedup="N/A"
            local webgpu_speedup="N/A"

            if [[ "$cpu_time" != "N/A" ]] && [[ "$mlx_time" != "N/A" ]]; then
                mlx_speedup=$(echo "scale=2; $cpu_time / $mlx_time" | bc 2>/dev/null || echo "N/A")
                mlx_speedup="${mlx_speedup}x"
            fi

            if [[ "$cpu_time" != "N/A" ]] && [[ "$webgpu_time" != "N/A" ]]; then
                webgpu_speedup=$(echo "scale=2; $cpu_time / $webgpu_time" | bc 2>/dev/null || echo "N/A")
                webgpu_speedup="${webgpu_speedup}x"
            fi

            echo "| $op | $cpu_time | $mlx_time | $webgpu_time | $mlx_speedup | $webgpu_speedup |" >> "$output_file"
        done
    else
        echo "| (install jq for detailed parsing) | | | | | |" >> "$output_file"
    fi

    cat >> "$output_file" << 'FOOTER'

## Notes

- CPU: Standard OpenFHE with OpenMP parallelization
- MLX: Metal-accelerated via Apple MLX framework
- WebGPU: Cross-platform GPU via WebGPU/Dawn

## Raw Results

See individual JSON files in the results directory for detailed metrics.
FOOTER

    log_success "Comparison report generated: $output_file"
    cat "$output_file"
}

# Main execution
main() {
    local run_cpu=0
    local run_mlx=0
    local run_webgpu=0

    # Parse arguments
    if [[ $# -eq 0 ]] || [[ "$1" == "--all" ]]; then
        run_cpu=1
        run_mlx=1
        run_webgpu=1
    else
        while [[ $# -gt 0 ]]; do
            case $1 in
                --cpu) run_cpu=1 ;;
                --mlx) run_mlx=1 ;;
                --webgpu) run_webgpu=1 ;;
                --help)
                    echo "Usage: $0 [--cpu] [--mlx] [--webgpu] [--all]"
                    exit 0
                    ;;
                *)
                    log_error "Unknown option: $1"
                    exit 1
                    ;;
            esac
            shift
        done
    fi

    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║           FHE Backend Comparison Benchmark Suite               ║"
    echo "║                 CPU vs MLX (Metal) vs WebGPU                   ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""

    check_backends

    # Run selected benchmarks
    if [[ $run_cpu -eq 1 ]] && [[ $CPU_AVAILABLE -eq 1 ]]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "                     CPU Backend Benchmarks                      "
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        run_benchmark "CPU" "$BUILD_CPU" "$RESULTS_DIR/cpu_${TIMESTAMP}.json"
        run_tfhe_benchmark "CPU" "$BUILD_CPU" "$RESULTS_DIR/cpu_tfhe_${TIMESTAMP}.json"
    fi

    if [[ $run_mlx -eq 1 ]] && [[ $MLX_AVAILABLE -eq 1 ]]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "                   MLX (Metal) Backend Benchmarks                "
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        run_benchmark "MLX" "$BUILD_MLX" "$RESULTS_DIR/mlx_${TIMESTAMP}.json"
        run_tfhe_benchmark "MLX" "$BUILD_MLX" "$RESULTS_DIR/mlx_tfhe_${TIMESTAMP}.json"
    fi

    if [[ $run_webgpu -eq 1 ]] && [[ $WEBGPU_AVAILABLE -eq 1 ]]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "                    WebGPU Backend Benchmarks                    "
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        run_benchmark "WebGPU" "$BUILD_WEBGPU" "$RESULTS_DIR/webgpu_${TIMESTAMP}.json"
        run_tfhe_benchmark "WebGPU" "$BUILD_WEBGPU" "$RESULTS_DIR/webgpu_tfhe_${TIMESTAMP}.json"
    fi

    # Generate comparison
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "                        Comparison Report                        "
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    generate_comparison

    echo ""
    log_success "All benchmarks completed!"
    echo ""
    echo "Results saved to: $RESULTS_DIR/"
    echo ""
}

main "$@"
