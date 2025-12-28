#!/bin/bash
# Quick benchmark using pre-built OpenFHE examples

set -e
cd "$(dirname "$0")/.."

echo "=========================================="
echo "OpenFHE Benchmark (using pre-built examples)"
echo "=========================================="
echo ""

EXAMPLES_DIR="build/bin/examples"

# Time a single run
time_example() {
    local name=$1
    local exe=$2
    echo -n "$name: "
    local start=$(python3 -c 'import time; print(time.time())')
    $exe > /dev/null 2>&1
    local end=$(python3 -c 'import time; print(time.time())')
    local elapsed=$(python3 -c "print(f'{($end - $start)*1000:.1f}')")
    echo "${elapsed} ms"
}

echo "=== BinFHE Examples ==="
time_example "boolean (GINX)" "$EXAMPLES_DIR/binfhe/boolean"
time_example "boolean-ap (AP)" "$EXAMPLES_DIR/binfhe/boolean-ap"
time_example "boolean-lmkcdey (LMKCDEY)" "$EXAMPLES_DIR/binfhe/boolean-lmkcdey"

echo ""
echo "=== PKE Examples ==="
time_example "simple-integers" "$EXAMPLES_DIR/pke/simple-integers"
time_example "simple-real-numbers (CKKS)" "$EXAMPLES_DIR/pke/simple-real-numbers"

echo ""
echo "=========================================="
echo "Done"
