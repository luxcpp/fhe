#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Lux Industries Inc
# CKKS Benchmark Comparison: Lux Lattice (Go) vs OpenFHE (C++)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo "CKKS Benchmark Comparison"
echo "Lux Lattice (Go) vs OpenFHE (C++)"
echo "=============================================="
echo ""

# ============================================
# Lux Lattice (Go) CKKS Benchmarks
# ============================================
echo "=============================================="
echo "1. Lux Lattice (Pure Go) CKKS"
echo "=============================================="

LATTICE_DIR="$HOME/work/lux/lattice"
if [ -d "$LATTICE_DIR" ]; then
    echo "Running Go CKKS benchmarks..."
    echo "(Parameters: LogN=14, N=16384 slots)"
    echo ""
    
    cd "$LATTICE_DIR"
    
    # Run benchmarks and capture output
    GO_RESULTS=$(go test -run=NONE -bench='BenchmarkCKKS/(Encoder|Evaluator/)' -benchtime=2s ./schemes/ckks/ 2>&1 | grep -E 'Benchmark|ns/op')
    
    echo "$GO_RESULTS" | tee "$RESULTS_DIR/go_ckks.txt"
    echo ""
    
    # Extract key metrics
    echo "--- Summary (Single-threaded) ---"
    echo "$GO_RESULTS" | grep -E 'Encoder/Encode.*Standard' | head -1 | awk '{printf "Encode:           %8.2f ms\n", $3/1000000}'
    echo "$GO_RESULTS" | grep -E 'Encoder/Decode.*Standard' | head -1 | awk '{printf "Decode:           %8.2f ms\n", $3/1000000}'
    echo "$GO_RESULTS" | grep -E 'Evaluator/Add/Ciphertext.*Standard' | head -1 | awk '{printf "Add Ciphertext:   %8.2f ms\n", $3/1000000}'
    echo "$GO_RESULTS" | grep -E 'Evaluator/Mul/Ciphertext.*Standard' | head -1 | awk '{printf "Mul Ciphertext:   %8.2f ms\n", $3/1000000}'
    echo "$GO_RESULTS" | grep -E 'Evaluator/MulRelin/.*Standard' | head -1 | awk '{printf "MulRelin:         %8.2f ms\n", $3/1000000}'
    echo "$GO_RESULTS" | grep -E 'Evaluator/Rescale.*Standard' | head -1 | awk '{printf "Rescale:          %8.2f ms\n", $3/1000000}'
    echo "$GO_RESULTS" | grep -E 'Evaluator/Rotate.*Standard' | head -1 | awk '{printf "Rotate:           %8.2f ms\n", $3/1000000}'
    
    echo ""
    echo "--- Summary (Parallel 10-core) ---"
    echo "$GO_RESULTS" | grep -E 'EvaluatorParallel/Add/Ciphertext' | head -1 | awk '{printf "Add Ciphertext:   %8.2f ms\n", $3/1000000}'
    echo "$GO_RESULTS" | grep -E 'EvaluatorParallel/Mul/Ciphertext' | head -1 | awk '{printf "Mul Ciphertext:   %8.2f ms\n", $3/1000000}'
    echo "$GO_RESULTS" | grep -E 'EvaluatorParallel/MulRelin/' | head -1 | awk '{printf "MulRelin:         %8.2f ms\n", $3/1000000}'
    echo "$GO_RESULTS" | grep -E 'EvaluatorParallel/Rescale' | head -1 | awk '{printf "Rescale:          %8.2f ms\n", $3/1000000}'
    echo "$GO_RESULTS" | grep -E 'EvaluatorParallel/Rotate' | head -1 | awk '{printf "Rotate:           %8.2f ms\n", $3/1000000}'
else
    echo "Error: Lattice directory not found at $LATTICE_DIR"
fi

echo ""

# ============================================
# OpenFHE CKKS Benchmarks (via examples)
# ============================================
echo "=============================================="
echo "2. OpenFHE (C++) CKKS"
echo "=============================================="

OPENFHE_DIR="$HOME/work/lux/fhe"
EXAMPLES_DIR="$OPENFHE_DIR/build/bin/examples/pke"

if [ -x "$EXAMPLES_DIR/simple-real-numbers" ]; then
    echo "Running OpenFHE CKKS examples with timing..."
    echo "(Ring dimension: 16384 = LogN=14)"
    echo ""
    
    cd "$OPENFHE_DIR"
    
    # Function to time multiple iterations
    time_iterations() {
        local name=$1
        local exe=$2
        local iters=${3:-5}
        
        local total=0
        for i in $(seq 1 $iters); do
            local start=$(python3 -c 'import time; print(time.time())')
            $exe > /dev/null 2>&1
            local end=$(python3 -c 'import time; print(time.time())')
            local elapsed=$(python3 -c "print($end - $start)")
            total=$(python3 -c "print($total + $elapsed)")
        done
        local avg=$(python3 -c "print(f'{($total / $iters)*1000:.2f}')")
        echo "$name: $avg ms (avg of $iters runs)"
    }
    
    # Time various CKKS examples
    echo "--- Full Example Timings ---"
    time_iterations "simple-real-numbers" "$EXAMPLES_DIR/simple-real-numbers" 5
    time_iterations "advanced-real-numbers" "$EXAMPLES_DIR/advanced-real-numbers" 3
    time_iterations "rotation" "$EXAMPLES_DIR/rotation" 3
    time_iterations "polynomial-evaluation" "$EXAMPLES_DIR/polynomial-evaluation" 3
    
    # More detailed timing using inner-product example
    echo ""
    echo "--- Per-Operation Estimates ---"
    echo "(Based on example runs - approximate values)"
    
    # Run advanced-real-numbers and capture timing
    # This example does: keygen, encrypt, add, sub, mult, rotate
    start=$(python3 -c 'import time; print(time.time())')
    "$EXAMPLES_DIR/advanced-real-numbers" > /dev/null 2>&1
    end=$(python3 -c 'import time; print(time.time())')
    total_ms=$(python3 -c "print(f'{($end - $start)*1000:.2f}')")
    echo "Total advanced-real-numbers: $total_ms ms"
    
else
    echo "Error: OpenFHE examples not found at $EXAMPLES_DIR"
    echo "Please build OpenFHE first: cd ~/work/lux/fhe && mkdir build && cd build && cmake .. && make -j"
fi

echo ""

# ============================================
# Comparison Summary
# ============================================
echo "=============================================="
echo "COMPARISON SUMMARY"
echo "=============================================="
echo ""
echo "| Implementation | Type | Notes |"
echo "|----------------|------|-------|"
echo "| Lux Lattice    | Pure Go | Full CKKS, parallel support, BSD-2-Clause |"
echo "| OpenFHE        | C++ | Industry standard, BSD-2-Clause |"
echo ""
echo "Both libraries are permissively licensed (BSD-2-Clause) and fully open source."
echo ""
echo "Recommendation:"
echo "- Use Lux Lattice for pure Go integration (no CGO needed)"
echo "- Use OpenFHE for C++ integration or maximum flexibility"
echo "- CGO bridge possible for Go → OpenFHE if needed"
echo ""
echo "=============================================="
echo "Benchmark Complete"
echo "=============================================="
