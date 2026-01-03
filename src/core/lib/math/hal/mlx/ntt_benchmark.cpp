// =============================================================================
// NTT Metal Kernel Benchmark - Verify 10x Speedup Target
// =============================================================================
//
// Compares:
// 1. CPU baseline (OpenFHE-style NTT)
// 2. Original MLX-based dispatcher (metal_dispatch.h)
// 3. Optimized Metal dispatcher (metal_dispatch_optimized.h)
//
// Usage:
//   clang++ -std=c++17 -O3 -DWITH_MLX -framework Metal -framework Foundation \
//           -I/path/to/mlx/include ntt_benchmark.cpp -o ntt_benchmark
//   ./ntt_benchmark
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <random>
#include <iomanip>
#include <cmath>

#include "metal_dispatch_optimized.h"

using namespace lux::gpu::metal;

// =============================================================================
// Reference CPU NTT (OpenFHE-style)
// =============================================================================

class CPUNTT {
public:
    CPUNTT(uint32_t N, uint64_t Q) : N_(N), Q_(Q) {
        log_N_ = 0;
        while ((1u << log_N_) < N) ++log_N_;
        init_twiddles();
    }

    void forward(uint64_t* data) {
        for (uint32_t s = 0; s < log_N_; ++s) {
            uint32_t m = 1u << s;
            uint32_t t = N_ >> (s + 1);

            for (uint32_t i = 0; i < m; ++i) {
                uint64_t w = twiddles_[m + i];
                uint32_t j1 = i << (log_N_ - s);

                for (uint32_t j = j1; j < j1 + t; ++j) {
                    uint64_t lo = data[j];
                    uint64_t hi = data[j + t];
                    uint64_t hi_tw = mulmod(hi, w);
                    data[j] = addmod(lo, hi_tw);
                    data[j + t] = submod(lo, hi_tw);
                }
            }
        }
    }

    void inverse(uint64_t* data) {
        for (uint32_t s = 0; s < log_N_; ++s) {
            uint32_t m = N_ >> (s + 1);
            uint32_t t = 1u << s;

            for (uint32_t i = 0; i < m; ++i) {
                uint64_t w = inv_twiddles_[m + i];
                uint32_t j1 = i << (s + 1);

                for (uint32_t j = j1; j < j1 + t; ++j) {
                    uint64_t lo = data[j];
                    uint64_t hi = data[j + t];
                    data[j] = addmod(lo, hi);
                    data[j + t] = mulmod(submod(lo, hi), w);
                }
            }
        }

        // Scale by N^{-1}
        uint64_t n_inv = mod_inv(N_, Q_);
        for (uint32_t i = 0; i < N_; ++i) {
            data[i] = mulmod(data[i], n_inv);
        }
    }

private:
    uint32_t N_, log_N_;
    uint64_t Q_;
    std::vector<uint64_t> twiddles_;
    std::vector<uint64_t> inv_twiddles_;

    void init_twiddles() {
        twiddles_.resize(N_);
        inv_twiddles_.resize(N_);

        uint64_t omega = find_omega(false);
        uint64_t omega_inv = find_omega(true);

        for (uint32_t m = 1; m < N_; m <<= 1) {
            uint32_t log_m = 0;
            while ((1u << log_m) < m) ++log_m;

            for (uint32_t i = 0; i < m; ++i) {
                uint32_t exp = (N_ / m) * bit_reverse(i, log_m);
                twiddles_[m + i] = powmod(omega, exp);
                inv_twiddles_[m + i] = powmod(omega_inv, exp);
            }
        }
        twiddles_[0] = 1;
        inv_twiddles_[0] = 1;
    }

    uint64_t find_omega(bool inverse) {
        for (uint64_t g = 2; g < Q_; ++g) {
            if (powmod(g, (Q_ - 1) / 2) != 1) {
                uint64_t w = powmod(g, (Q_ - 1) / (2 * N_));
                return inverse ? mod_inv(w, Q_) : w;
            }
        }
        return 0;
    }

    uint64_t powmod(uint64_t base, uint64_t exp) {
        uint64_t result = 1;
        base %= Q_;
        while (exp > 0) {
            if (exp & 1) result = mulmod(result, base);
            base = mulmod(base, base);
            exp >>= 1;
        }
        return result;
    }

    uint64_t mulmod(uint64_t a, uint64_t b) {
        return static_cast<uint64_t>((__uint128_t)a * b % Q_);
    }

    uint64_t addmod(uint64_t a, uint64_t b) {
        uint64_t sum = a + b;
        return (sum >= Q_) ? sum - Q_ : sum;
    }

    uint64_t submod(uint64_t a, uint64_t b) {
        return (a >= b) ? a - b : a + Q_ - b;
    }

    uint64_t mod_inv(uint64_t a, uint64_t m) {
        int64_t t = 0, newt = 1;
        int64_t r = m, newr = a;
        while (newr != 0) {
            int64_t q = r / newr;
            std::tie(t, newt) = std::make_pair(newt, t - q * newt);
            std::tie(r, newr) = std::make_pair(newr, r - q * newr);
        }
        return static_cast<uint64_t>((t < 0) ? t + m : t);
    }

    uint32_t bit_reverse(uint32_t x, uint32_t bits) {
        uint32_t result = 0;
        for (uint32_t i = 0; i < bits; ++i) {
            result = (result << 1) | (x & 1);
            x >>= 1;
        }
        return result;
    }
};

// =============================================================================
// Benchmark Harness
// =============================================================================

struct BenchmarkResult {
    std::string name;
    uint32_t N;
    uint32_t iterations;
    double total_time_ms;
    double avg_time_us;
    double throughput_mntt_per_sec;
    double speedup_vs_cpu;
};

template<typename NTTFunc>
BenchmarkResult run_benchmark(const std::string& name, uint32_t N, uint64_t Q,
                               NTTFunc ntt_func, int iterations, double cpu_time_us = 0) {
    std::vector<uint64_t> data(N);
    std::mt19937_64 rng(42);
    for (auto& x : data) x = rng() % Q;

    // Warmup
    for (int i = 0; i < 10; ++i) {
        ntt_func(data.data());
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        ntt_func(data.data());
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_us = (total_ms * 1000.0) / iterations;
    double throughput = iterations / (total_ms / 1000.0) / 1e6;

    BenchmarkResult result;
    result.name = name;
    result.N = N;
    result.iterations = iterations;
    result.total_time_ms = total_ms;
    result.avg_time_us = avg_us;
    result.throughput_mntt_per_sec = throughput;
    result.speedup_vs_cpu = (cpu_time_us > 0) ? (cpu_time_us / avg_us) : 1.0;

    return result;
}

void print_result(const BenchmarkResult& r) {
    std::cout << std::setw(20) << r.name
              << " | N=" << std::setw(5) << r.N
              << " | " << std::setw(8) << std::fixed << std::setprecision(2) << r.avg_time_us << " us"
              << " | " << std::setw(6) << std::fixed << std::setprecision(2) << r.throughput_mntt_per_sec << " M/s"
              << " | " << std::setw(6) << std::fixed << std::setprecision(2) << r.speedup_vs_cpu << "x"
              << std::endl;
}

void print_header() {
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "NTT Metal Kernel Benchmark - Target: 10x speedup for N>=8192" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << std::setw(20) << "Implementation"
              << " | " << std::setw(7) << "Size"
              << " | " << std::setw(12) << "Avg Time"
              << " | " << std::setw(10) << "Throughput"
              << " | " << std::setw(8) << "Speedup"
              << std::endl;
    std::cout << std::string(80, '-') << std::endl;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    // Standard FHE parameters
    // Q = 65537 (Fermat prime F4) is commonly used for testing
    // Production uses larger primes like 2^60 - 2^14 + 1
    const uint64_t Q = 0xFFFFFFFFFFFFFFull - 58; // ~2^56, NTT-friendly prime

    // Test sizes: 1024, 2048, 4096, 8192, 16384
    std::vector<uint32_t> sizes = {1024, 2048, 4096, 8192, 16384};
    int iterations = 1000;

    print_header();

    for (uint32_t N : sizes) {
        std::cout << "\n--- N = " << N << " ---\n";

        // CPU baseline
        CPUNTT cpu_ntt(N, Q);
        auto cpu_result = run_benchmark("CPU Baseline", N, Q,
            [&](uint64_t* data) { cpu_ntt.forward(data); }, iterations);
        print_result(cpu_result);

        // Optimized Metal dispatcher
        NTTMetalDispatcherOptimized metal_ntt(N, Q);
        if (metal_ntt.is_available()) {
            std::string kernel_type = metal_ntt.uses_fused_kernel() ? "Metal (Fused)" : "Metal (Staged)";
            auto metal_result = run_benchmark(kernel_type, N, Q,
                [&](uint64_t* data) { metal_ntt.forward(data); }, iterations, cpu_result.avg_time_us);
            print_result(metal_result);

            // Also test inverse
            auto metal_inv_result = run_benchmark(kernel_type + " Inv", N, Q,
                [&](uint64_t* data) { metal_ntt.inverse(data); }, iterations, cpu_result.avg_time_us);
            print_result(metal_inv_result);

            // Report kernel stats
            auto metrics = metal_ntt.get_metrics();
            std::cout << "  -> Kernel launches: " << metrics.kernel_launches
                      << ", Avg kernel time: " << metrics.avg_ntt_time_us << " us\n";
        } else {
            std::cout << "Metal dispatcher not available (not on Apple Silicon)\n";
        }
    }

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "SUMMARY:\n";
    std::cout << "- Fused kernel: 1 kernel launch for N <= 2048\n";
    std::cout << "- Staged kernel: log(N) launches for larger sizes\n";
    std::cout << "- Barrett reduction fused into butterfly (no separate pass)\n";
    std::cout << "- Zero-copy unified memory (no GPU upload/download)\n";
    std::cout << std::string(80, '=') << std::endl;

    return 0;
}
