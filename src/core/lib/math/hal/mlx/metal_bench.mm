// =============================================================================
// Native Metal NTT Benchmark
// =============================================================================
// Compile:
//   clang++ -std=c++17 -O3 -x objective-c++ -fobjc-arc \
//     -framework Metal -framework Foundation \
//     -I../../include -I../../lib \
//     metal_bench.mm -o metal_bench
//
// Run:
//   ./metal_bench
// =============================================================================

#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstdlib>

#include "metal_dispatch_optimized.h"

using namespace lux::gpu::metal;

// Time a function in microseconds
template<typename Func>
double benchmark_us(Func&& fn, int warmup = 5, int iters = 20) {
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        fn();
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        fn();
    }
    auto end = std::chrono::high_resolution_clock::now();
    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    return total_us / iters;
}

void print_header() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           Lux FHE Native Metal NTT Benchmark                         ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  NTT Implementation: Fused Metal Kernel with Shared Memory           ║\n";
    std::cout << "║  Backend: Native Metal API (Apple Silicon GPU)                       ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
}

// CPU reference NTT for comparison
void ntt_cpu(uint64_t* data, uint32_t N, uint64_t Q, const std::vector<uint64_t>& tw) {
    uint32_t log_N = 0;
    while ((1u << log_N) < N) ++log_N;

    for (uint32_t s = 0; s < log_N; ++s) {
        uint32_t m = 1u << s;
        uint32_t t = N >> (s + 1);

        for (uint32_t i = 0; i < m; ++i) {
            uint32_t j1 = i << (log_N - s);
            uint32_t j2 = j1 + t;
            uint64_t w = tw[m + i];

            for (uint32_t j = j1; j < j2; ++j) {
                uint64_t lo = data[j];
                uint64_t hi = data[j + t];
                uint64_t whi = (__uint128_t)hi * w % Q;
                data[j] = (lo + whi) % Q;
                data[j + t] = (lo >= whi) ? (lo - whi) : (lo + Q - whi);
            }
        }
    }
}

int main() {
    print_header();

    // NTT prime: q = 119 * 2^23 + 1
    uint64_t Q = 998244353;

    std::vector<uint32_t> sizes = {1024, 2048, 4096, 8192, 16384};
    std::vector<uint32_t> batches = {1, 8, 32, 128};

    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                   CPU vs Native Metal GPU                            ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << std::left << std::setw(8) << "N"
              << std::setw(6) << "B"
              << std::setw(14) << "CPU (µs)"
              << std::setw(14) << "GPU (µs)"
              << std::setw(10) << "Speedup"
              << std::endl;
    std::cout << std::string(52, '-') << std::endl;

    for (uint32_t N : sizes) {
        NTTMetalDispatcherOptimized gpu_ntt(N, Q);

        if (!gpu_ntt.is_available()) {
            std::cout << "N=" << N << ": Metal not available\n";
            continue;
        }

        std::cout << "N=" << N << " (Fused: " << (gpu_ntt.uses_fused_kernel() ? "yes" : "no") << ")\n";

        // Precompute CPU twiddles
        auto powmod = [](uint64_t base, uint64_t exp, uint64_t m) -> uint64_t {
            uint64_t result = 1;
            base %= m;
            while (exp > 0) {
                if (exp & 1) result = (__uint128_t)result * base % m;
                base = (__uint128_t)base * base % m;
                exp >>= 1;
            }
            return result;
        };

        auto bit_reverse = [](uint32_t x, uint32_t bits) -> uint32_t {
            uint32_t result = 0;
            for (uint32_t i = 0; i < bits; ++i) {
                result = (result << 1) | (x & 1);
                x >>= 1;
            }
            return result;
        };

        uint64_t omega = 0;
        for (uint64_t g = 2; g < Q; ++g) {
            if (powmod(g, (Q - 1) / 2, Q) != 1) {
                omega = powmod(g, (Q - 1) / (2 * N), Q);
                break;
            }
        }

        std::vector<uint64_t> tw(N);
        uint32_t log_N = 0;
        while ((1u << log_N) < N) ++log_N;

        for (uint32_t m = 1; m < N; m <<= 1) {
            uint32_t log_m = 0;
            while ((1u << log_m) < m) ++log_m;
            for (uint32_t i = 0; i < m; ++i) {
                uint32_t exp = (N / m) * bit_reverse(i, log_m);
                tw[m + i] = powmod(omega, exp, Q);
            }
        }
        tw[0] = 1;

        for (uint32_t batch : batches) {
            // Create test data
            std::vector<uint64_t> cpu_data(N * batch);
            std::vector<uint64_t> gpu_data(N * batch);
            for (size_t i = 0; i < cpu_data.size(); ++i) {
                uint64_t val = static_cast<uint64_t>(rand()) % Q;
                cpu_data[i] = val;
                gpu_data[i] = val;
            }

            // CPU benchmark
            double cpu_us = benchmark_us([&]() {
                for (uint32_t b = 0; b < batch; ++b) {
                    std::vector<uint64_t> tmp(cpu_data.begin() + b * N,
                                               cpu_data.begin() + (b + 1) * N);
                    ntt_cpu(tmp.data(), N, Q, tw);
                }
            }, 3, 10);

            // GPU benchmark
            double gpu_us = benchmark_us([&]() {
                std::vector<uint64_t> tmp = gpu_data;
                gpu_ntt.forward(tmp.data(), batch);
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
        std::cout << "\n";
    }

    std::cout << "=== Benchmark Complete ===\n";

    return 0;
}
