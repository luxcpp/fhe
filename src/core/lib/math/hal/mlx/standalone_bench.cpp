// =============================================================================
// Standalone Metal NTT Benchmark
// =============================================================================
//
// Compile:
//   clang++ -std=c++17 -O3 -DWITH_MLX \
//     -I../../include -I../../lib \
//     -I../../../../../.venv/lib/python3.12/site-packages/mlx/include \
//     -L../../../../../.venv/lib/python3.12/site-packages/mlx/lib -lmlx \
//     -framework Metal -framework Foundation \
//     -Wl,-rpath,../../../../../.venv/lib/python3.12/site-packages/mlx/lib \
//     standalone_bench.cpp -o bench
//
// Run:
//   ./bench
//
// =============================================================================

#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <cmath>

#ifdef WITH_MLX
#include <mlx/mlx.h>
namespace mx = mlx::core;
#endif

#include "ntt.h"
#include "ntt_fourstep.h"

using namespace lbcrypto::gpu;

// Benchmark result structure
struct BenchResult {
    std::string name;
    uint32_t N;
    uint32_t batch;
    double avg_us;
    double throughput_gops;
    double bandwidth_gbps;
};

// Time a function in microseconds
template<typename Func>
double benchmark_us(Func&& fn, int warmup = 5, int iters = 20) {
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        fn();
    }

#ifdef WITH_MLX
    mx::synchronize();
#endif

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        fn();
    }

#ifdef WITH_MLX
    mx::synchronize();
#endif

    auto end = std::chrono::high_resolution_clock::now();
    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    return total_us / iters;
}

void print_header() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                   Lux FHE Metal GPU Benchmark                        ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  NTT Implementation: Four-Step with Unified Memory                   ║\n";
    std::cout << "║  Backend: MLX Metal (Apple Silicon GPU)                              ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
}

void print_result(const BenchResult& r) {
    std::cout << std::left << std::setw(25) << r.name
              << " N=" << std::setw(6) << r.N
              << " B=" << std::setw(4) << r.batch
              << " | " << std::right << std::setw(8) << std::fixed << std::setprecision(1)
              << r.avg_us << " µs"
              << " | " << std::setw(6) << std::setprecision(2) << r.throughput_gops << " Gops"
              << " | " << std::setw(6) << std::setprecision(1) << r.bandwidth_gbps << " GB/s"
              << std::endl;
}

void run_cpu_vs_gpu_benchmark() {
#ifdef WITH_MLX
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                      CPU vs GPU COMPARISON                           ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n\n";

    uint64_t Q = 998244353;
    std::vector<uint32_t> sizes = {1024, 2048, 4096, 8192, 16384};
    std::vector<uint32_t> batches = {1, 8, 32};

    std::cout << std::left << std::setw(8) << "N"
              << std::setw(6) << "B"
              << std::setw(14) << "CPU (µs)"
              << std::setw(14) << "GPU (µs)"
              << std::setw(10) << "Speedup"
              << std::endl;
    std::cout << std::string(52, '-') << std::endl;

    for (uint32_t N : sizes) {
        NTTEngine engine(N, Q);

        for (uint32_t batch : batches) {
            // Create test data
            std::vector<int64_t> data(N * batch);
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = static_cast<int64_t>(rand() % Q);
            }

            // CPU benchmark
            mx::set_default_device(mx::Device::cpu);
            mx::array cpu_input(data.data(), {static_cast<int>(batch), static_cast<int>(N)}, mx::int64);
            mx::eval(cpu_input);

            double cpu_us = benchmark_us([&]() {
                mx::array tmp = cpu_input;
                engine.forward(tmp);
                mx::eval(tmp);
            }, 3, 10);

            // GPU benchmark
            mx::set_default_device(mx::Device::gpu);
            mx::array gpu_input(data.data(), {static_cast<int>(batch), static_cast<int>(N)}, mx::int64);
            mx::eval(gpu_input);

            double gpu_us = benchmark_us([&]() {
                mx::array tmp = gpu_input;
                engine.forward(tmp);
                mx::eval(tmp);
            }, 5, 20);

            double speedup = cpu_us / gpu_us;

            std::cout << std::left << std::setw(8) << N
                      << std::setw(6) << batch
                      << std::setw(14) << std::fixed << std::setprecision(1) << cpu_us
                      << std::setw(14) << gpu_us
                      << std::setw(10) << std::setprecision(2) << speedup << "x";

            if (speedup > 2.0) std::cout << " 🚀";
            if (speedup > 5.0) std::cout << "🔥";
            if (speedup > 10.0) std::cout << "⚡";
            std::cout << std::endl;
        }
    }

    // Reset to GPU
    mx::set_default_device(mx::Device::gpu);
#endif
}

int main() {
    print_header();

#ifdef WITH_MLX
    std::cout << "Metal GPU: " << (mx::metal::is_available() ? "Available ✓" : "Not Available ✗") << "\n";
    if (mx::metal::is_available()) {
        mx::set_default_device(mx::Device::gpu);
        std::cout << "Default device: GPU\n";
    }
#else
    std::cout << "MLX not enabled - CPU only\n";
#endif

    // Run CPU vs GPU comparison first
    run_cpu_vs_gpu_benchmark();

    std::cout << "\n=== NTT Forward/Inverse Benchmarks (GPU) ===\n\n";

    std::vector<uint32_t> sizes = {1024, 2048, 4096, 8192, 16384};
    std::vector<uint32_t> batches = {1, 8, 32, 128};

    // Use GPU-friendly NTT prime: q = k * 2^23 + 1
    uint64_t Q = 998244353;  // 119 * 2^23 + 1, primitive root = 3

    for (uint32_t N : sizes) {
        std::cout << "--- N = " << N << " ---\n";

        NTTEngine engine(N, Q);

        for (uint32_t batch : batches) {
#ifdef WITH_MLX
            // Create random input data
            std::vector<int64_t> data(N * batch);
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = static_cast<int64_t>(rand() % Q);
            }

            mx::array input(data.data(), {static_cast<int>(batch), static_cast<int>(N)}, mx::int64);
            mx::eval(input);

            // Benchmark forward NTT
            double fwd_us = benchmark_us([&]() {
                mx::array tmp = input;
                engine.forward(tmp);
                mx::eval(tmp);
            });

            // Benchmark inverse NTT
            double inv_us = benchmark_us([&]() {
                mx::array tmp = input;
                engine.inverse(tmp);
                mx::eval(tmp);
            });

            // Calculate metrics
            // NTT is O(N log N) butterflies, each butterfly = 2 muls + 2 adds
            double ops_per_ntt = N * std::log2(N) * 4;
            double total_ops = ops_per_ntt * batch;

            // Bandwidth: read N elements, write N elements, per poly
            double bytes_per_ntt = N * 8 * 2;  // 8 bytes per int64, read + write
            double total_bytes = bytes_per_ntt * batch;

            BenchResult fwd_result;
            fwd_result.name = "NTT Forward";
            fwd_result.N = N;
            fwd_result.batch = batch;
            fwd_result.avg_us = fwd_us;
            fwd_result.throughput_gops = (total_ops / fwd_us) / 1000.0;  // Gops
            fwd_result.bandwidth_gbps = (total_bytes / fwd_us) / 1000.0;  // GB/s

            BenchResult inv_result;
            inv_result.name = "NTT Inverse";
            inv_result.N = N;
            inv_result.batch = batch;
            inv_result.avg_us = inv_us;
            inv_result.throughput_gops = (total_ops / inv_us) / 1000.0;
            inv_result.bandwidth_gbps = (total_bytes / inv_us) / 1000.0;

            print_result(fwd_result);
            print_result(inv_result);
#else
            std::cout << "  Batch " << batch << ": (MLX required)\n";
#endif
        }
        std::cout << "\n";
    }

    // Four-Step NTT benchmark for large sizes
    std::cout << "=== Four-Step NTT (Large N) ===\n\n";

    for (uint32_t N : {16384u, 32768u, 65536u}) {
#ifdef WITH_MLX
        NTTFourStep fourStep(N, Q);

        std::vector<int64_t> data(N);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<int64_t>(rand() % Q);
        }

        mx::array input(data.data(), {static_cast<int>(N)}, mx::int64);
        mx::eval(input);

        double us = benchmark_us([&]() {
            mx::array tmp = input;
            fourStep.forward(tmp);
            mx::eval(tmp);
        });

        double ops = N * std::log2(N) * 4;
        double bytes = N * 8 * 2;

        std::cout << "Four-Step N=" << std::setw(6) << N
                  << " | " << std::setw(8) << std::fixed << std::setprecision(1)
                  << us << " µs"
                  << " | " << std::setw(6) << std::setprecision(2)
                  << (ops / us) / 1000.0 << " Gops"
                  << " | " << std::setw(6) << std::setprecision(1)
                  << (bytes / us) / 1000.0 << " GB/s"
                  << std::endl;
#else
        std::cout << "N=" << N << ": (MLX required)\n";
#endif
    }

    std::cout << "\n=== Benchmark Complete ===\n";

    return 0;
}
