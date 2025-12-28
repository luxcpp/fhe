#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Lux Industries Inc
#
# Run FHE benchmarks for luxfi/fhe (OpenFHE) and luxfi/lattice (Go)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "FHE Benchmark Suite"
echo "=========================================="
echo "Timestamp: $TIMESTAMP"
echo ""

# Build and run OpenFHE benchmarks
run_openfhe() {
    echo "=== Building OpenFHE Benchmarks ==="
    cd "$SCRIPT_DIR/openfhe"
    
    if [ ! -d "build" ]; then
        mkdir build
    fi
    cd build
    
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DOpenFHE_DIR="$SCRIPT_DIR/../build/lib/cmake/OpenFHE"
    make -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)
    
    echo ""
    echo "=== Running OpenFHE BinFHE Benchmarks ==="
    ./binfhe_bench 2>&1 | tee "$RESULTS_DIR/openfhe_binfhe_$TIMESTAMP.txt"
    
    echo ""
    echo "=== Running OpenFHE PKE Benchmarks ==="
    ./pke_bench 2>&1 | tee "$RESULTS_DIR/openfhe_pke_$TIMESTAMP.txt"
}

# Run Lattice (Go) benchmarks
run_lattice() {
    echo ""
    echo "=== Running Lattice (Go) CKKS Benchmarks ==="
    
    LATTICE_DIR="$HOME/work/lux/lattice"
    if [ -d "$LATTICE_DIR" ]; then
        cd "$LATTICE_DIR"
        go test -run=NONE -bench='BenchmarkCKKS' -benchtime=2s ./schemes/ckks/ 2>&1 | \
            tee "$RESULTS_DIR/lattice_ckks_$TIMESTAMP.txt"
    else
        echo "Error: Lattice directory not found at $LATTICE_DIR"
    fi
}

# Generate summary report
generate_report() {
    echo ""
    echo "=== Generating Benchmark Summary ==="
    
    python3 << EOF
import re
import sys

def parse_openfhe(filename):
    results = {}
    try:
        with open(filename) as f:
            content = f.read()
            # Parse lines like "AND: 12.345 ms"
            for line in content.split('\n'):
                match = re.match(r'^(\w+):\s+([\d.]+)\s+ms', line.strip())
                if match:
                    results[match.group(1)] = float(match.group(2))
    except:
        pass
    return results

def parse_lattice(filename):
    results = {}
    try:
        with open(filename) as f:
            content = f.read()
            # Parse Go benchmark output
            for line in content.split('\n'):
                match = re.match(r'Benchmark(\w+).*\s+([\d.]+)\s+ns/op', line.strip())
                if match:
                    name = match.group(1)
                    ns = float(match.group(2))
                    results[name] = ns / 1000000  # Convert to ms
    except:
        pass
    return results

print("\n" + "=" * 60)
print("FHE BENCHMARK SUMMARY")
print("=" * 60)

openfhe = parse_openfhe('$RESULTS_DIR/openfhe_binfhe_$TIMESTAMP.txt')
lattice = parse_lattice('$RESULTS_DIR/lattice_ckks_$TIMESTAMP.txt')

if openfhe:
    print(f"\nOpenFHE BinFHE (Boolean Operations):")
    print("-" * 40)
    for op, time in sorted(openfhe.items()):
        print(f"  {op:<20} {time:.3f} ms")

if lattice:
    print(f"\nLattice CKKS (Approximate Arithmetic):")
    print("-" * 40)
    for op, time in sorted(lattice.items())[:10]:  # Top 10
        print(f"  {op:<30} {time:.3f} ms")

print("\n" + "=" * 60)
print("All results use permissively licensed (BSD-2-Clause) libraries.")
print("=" * 60)
EOF
}

# Main
case "${1:-all}" in
    openfhe)
        run_openfhe
        ;;
    lattice)
        run_lattice
        ;;
    report)
        generate_report
        ;;
    all)
        run_openfhe
        run_lattice
        generate_report
        ;;
    *)
        echo "Usage: $0 [openfhe|lattice|report|all]"
        exit 1
        ;;
esac

echo ""
echo "Results saved to: $RESULTS_DIR"
