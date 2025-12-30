#!/bin/bash
# Copyright (C) 2024-2025 Lux Industries Inc.
# SPDX-License-Identifier: Apache-2.0
#
# FHE Server Benchmark Comparison: Go vs C++

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FHE_DIR="${SCRIPT_DIR}/.."
GO_SERVER_DIR="${HOME}/work/lux/fhe"
CPP_SERVER_DIR="${FHE_DIR}"

# Configuration
GO_PORT=8080
CPP_PORT=8081
NUM_REQUESTS=1000
CONCURRENCY=10
WARMUP_REQUESTS=100

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         FHE Server Benchmark: Go vs C++                       ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check if servers are already running
check_port() {
    local port=$1
    if lsof -i :$port > /dev/null 2>&1; then
        return 0
    fi
    return 1
}

# Start Go server
start_go_server() {
    echo -e "${YELLOW}Starting Go FHE server on port $GO_PORT...${NC}"
    
    if check_port $GO_PORT; then
        echo -e "${GREEN}Go server already running on port $GO_PORT${NC}"
        return
    fi
    
    cd "$GO_SERVER_DIR"
    
    if [ ! -f "go.mod" ]; then
        echo -e "${RED}Go server not found at $GO_SERVER_DIR${NC}"
        return 1
    fi
    
    # Build if needed
    if [ ! -f "./server/server" ]; then
        echo "Building Go server..."
        go build -o ./server/server ./server/
    fi
    
    ./server/server -port $GO_PORT &
    GO_PID=$!
    sleep 2
    
    if ! check_port $GO_PORT; then
        echo -e "${RED}Failed to start Go server${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Go server started (PID: $GO_PID)${NC}"
}

# Start C++ server
start_cpp_server() {
    echo -e "${YELLOW}Starting C++ FHE server on port $CPP_PORT...${NC}"
    
    if check_port $CPP_PORT; then
        echo -e "${GREEN}C++ server already running on port $CPP_PORT${NC}"
        return
    fi
    
    local cpp_binary="${CPP_SERVER_DIR}/build/src/gpu/lux_fhe_server"
    
    if [ ! -f "$cpp_binary" ]; then
        echo "Building C++ server..."
        mkdir -p "${CPP_SERVER_DIR}/build"
        cd "${CPP_SERVER_DIR}/build"
        cmake .. -DBUILD_SERVER=ON -DWITH_CUDA=OFF
        make -j$(nproc) lux_fhe_server 2>/dev/null || make lux_fhe_server
        cd -
    fi
    
    if [ ! -f "$cpp_binary" ]; then
        echo -e "${RED}C++ server not built. Run cmake && make first.${NC}"
        return 1
    fi
    
    "$cpp_binary" --port $CPP_PORT &
    CPP_PID=$!
    sleep 2
    
    if ! check_port $CPP_PORT; then
        echo -e "${RED}Failed to start C++ server${NC}"
        return 1
    fi
    
    echo -e "${GREEN}C++ server started (PID: $CPP_PID)${NC}"
}

# Cleanup
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    if [ ! -z "$GO_PID" ]; then
        kill $GO_PID 2>/dev/null || true
    fi
    if [ ! -z "$CPP_PID" ]; then
        kill $CPP_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Run benchmark against a server
benchmark_server() {
    local name=$1
    local port=$2
    local endpoint=$3
    local data=$4
    
    echo -e "\n${YELLOW}Benchmarking $name - $endpoint${NC}"
    
    # Check if server is healthy
    if ! curl -s "http://localhost:$port/health" > /dev/null; then
        echo -e "${RED}Server not responding on port $port${NC}"
        return 1
    fi
    
    # Warmup
    echo "  Warming up ($WARMUP_REQUESTS requests)..."
    for i in $(seq 1 $WARMUP_REQUESTS); do
        curl -s -X POST "http://localhost:$port$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data" > /dev/null
    done
    
    # Benchmark with hey or ab
    if command -v hey &> /dev/null; then
        echo "  Running benchmark ($NUM_REQUESTS requests, $CONCURRENCY concurrent)..."
        hey -n $NUM_REQUESTS -c $CONCURRENCY -m POST \
            -H "Content-Type: application/json" \
            -d "$data" \
            "http://localhost:$port$endpoint"
    elif command -v ab &> /dev/null; then
        echo "  Running benchmark with Apache Bench..."
        echo "$data" > /tmp/bench_data.json
        ab -n $NUM_REQUESTS -c $CONCURRENCY -p /tmp/bench_data.json \
            -T "application/json" \
            "http://localhost:$port$endpoint"
    else
        echo "  Running basic benchmark..."
        local start=$(date +%s.%N)
        for i in $(seq 1 $NUM_REQUESTS); do
            curl -s -X POST "http://localhost:$port$endpoint" \
                -H "Content-Type: application/json" \
                -d "$data" > /dev/null
        done
        local end=$(date +%s.%N)
        local duration=$(echo "$end - $start" | bc)
        local rps=$(echo "$NUM_REQUESTS / $duration" | bc -l)
        echo "  Total time: ${duration}s"
        echo "  Requests/sec: ${rps}"
    fi
}

# Main benchmark sequence
run_benchmarks() {
    echo -e "\n${GREEN}=== Starting Benchmarks ===${NC}\n"
    
    # Generate key request
    local gen_key_data='{"bit_width": 64}'
    
    # After generating key, use it for encrypt/evaluate
    local encrypt_data='{"key_id": "test-key", "value": 42, "bit_width": 64}'
    local evaluate_data='{"key_id": "test-key", "operation": "add", "operands": ["Y2lwaGVydGV4dDE=", "Y2lwaGVydGV4dDI="]}'
    local bootstrap_data='{"key_id": "test-key", "ciphertext": "Y2lwaGVydGV4dA=="}'
    
    # First generate a key on each server
    echo -e "${YELLOW}Generating keys...${NC}"
    curl -s -X POST "http://localhost:$GO_PORT/publickey/generate" \
        -H "Content-Type: application/json" -d "$gen_key_data" > /dev/null || true
    curl -s -X POST "http://localhost:$CPP_PORT/publickey/generate" \
        -H "Content-Type: application/json" -d "$gen_key_data" > /dev/null || true
    
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "                      HEALTH CHECK"
    echo "═══════════════════════════════════════════════════════════════"
    benchmark_server "Go" $GO_PORT "/health" "{}"
    benchmark_server "C++" $CPP_PORT "/health" "{}"
    
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "                      ENCRYPT"
    echo "═══════════════════════════════════════════════════════════════"
    benchmark_server "Go" $GO_PORT "/encrypt" "$encrypt_data"
    benchmark_server "C++" $CPP_PORT "/encrypt" "$encrypt_data"
    
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "                      EVALUATE (ADD)"
    echo "═══════════════════════════════════════════════════════════════"
    benchmark_server "Go" $GO_PORT "/evaluate" "$evaluate_data"
    benchmark_server "C++" $CPP_PORT "/evaluate" "$evaluate_data"
    
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "                      BOOTSTRAP"
    echo "═══════════════════════════════════════════════════════════════"
    benchmark_server "Go" $GO_PORT "/bootstrap" "$bootstrap_data"
    benchmark_server "C++" $CPP_PORT "/bootstrap" "$bootstrap_data"
    
    echo ""
    echo -e "${GREEN}=== Benchmarks Complete ===${NC}"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --go-only)
            GO_ONLY=true
            shift
            ;;
        --cpp-only)
            CPP_ONLY=true
            shift
            ;;
        --requests)
            NUM_REQUESTS=$2
            shift 2
            ;;
        --concurrency)
            CONCURRENCY=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Start servers
if [ -z "$CPP_ONLY" ]; then
    start_go_server || echo "Skipping Go server"
fi

if [ -z "$GO_ONLY" ]; then
    start_cpp_server || echo "Skipping C++ server"
fi

# Run benchmarks
run_benchmarks

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                    Benchmark Complete                         ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
