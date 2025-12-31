// =============================================================================
// NTT Batch Optimization Benchmark
// =============================================================================
//
// Benchmark suite comparing:
// - Original sequential batch processing
// - Optimized vectorized batch processing
//
// Expected speedup: 2-3x additional on top of existing 6.48x
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

#ifndef LBCRYPTO_MATH_HAL_MLX_NTT_BATCH_BENCHMARK_H
#define LBCRYPTO_MATH_HAL_MLX_NTT_BATCH_BENCHMARK_H

#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>

#ifdef WITH_MLX
#include <mlx/mlx.h>
#include "ntt_batch_optimized.h"
#include "metal_dispatch.h"
namespace mx = mlx::core;
#endif

namespace lbcrypto {
namespace gpu {
namespace metal {

#ifdef WITH_MLX

// =============================================================================
// Benchmark Result
// =============================================================================

struct NTTBenchmarkResult {
    double forward_time_ms;
    double inverse_time_ms;
    double poly_mul_time_ms;
    int batch_size;
    int ring_dim;
    bool correct;

    void print(const char* label) const {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << label << ": "
                  << "N=" << ring_dim << ", B=" << batch_size
                  << " | fwd=" << forward_time_ms << "ms"
                  << " | inv=" << inverse_time_ms << "ms"
                  << " | mul=" << poly_mul_time_ms << "ms"
                  << " | " << (correct ? "PASS" : "FAIL")
                  << std::endl;
    }
};

// =============================================================================
// Benchmark Functions
// =============================================================================

inline NTTBenchmarkResult benchmark_batch_vectorized(
    uint32_t N, uint64_t Q, int batch_size, int iterations = 100) {

    NTTBenchmarkResult result;
    result.ring_dim = N;
    result.batch_size = batch_size;

    // Create optimized engine
    BatchVectorizedNTT engine(N, Q);

    if (!engine.is_gpu_available()) {
        result.correct = false;
        result.forward_time_ms = -1;
        result.inverse_time_ms = -1;
        result.poly_mul_time_ms = -1;
        return result;
    }

    // Generate random test data
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int64_t> dist(0, Q - 1);

    std::vector<int64_t> data(batch_size * N);
    for (auto& v : data) {
        v = dist(rng);
    }

    auto input = mx::array(data.data(), {batch_size, static_cast<int>(N)}, mx::int64);
    mx::eval(input);

    // Warm-up
    for (int i = 0; i < 10; ++i) {
        auto tmp = mx::array(input);
        engine.forward(tmp);
        mx::eval(tmp);
    }

    // Benchmark forward NTT
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto tmp = mx::array(input);
        engine.forward(tmp);
        mx::eval(tmp);
    }
    auto end = std::chrono::high_resolution_clock::now();
    result.forward_time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

    // Benchmark inverse NTT
    auto ntt_data = mx::array(input);
    engine.forward(ntt_data);
    mx::eval(ntt_data);

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto tmp = mx::array(ntt_data);
        engine.inverse(tmp);
        mx::eval(tmp);
    }
    end = std::chrono::high_resolution_clock::now();
    result.inverse_time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

    // Benchmark polynomial multiply
    auto a = mx::array(input);
    auto b = mx::array(input);

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto prod = engine.poly_mul(a, b);
        mx::eval(prod);
    }
    end = std::chrono::high_resolution_clock::now();
    result.poly_mul_time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

    // Verify correctness: forward then inverse should give original
    auto test = mx::array(input);
    engine.forward(test);
    engine.inverse(test);
    mx::eval(test);

    auto test_ptr = test.data<int64_t>();
    auto orig_ptr = input.data<int64_t>();
    result.correct = true;
    for (int i = 0; i < batch_size * static_cast<int>(N); ++i) {
        if (test_ptr[i] != orig_ptr[i]) {
            result.correct = false;
            break;
        }
    }

    return result;
}

// Benchmark standard NTT for comparison
inline NTTBenchmarkResult benchmark_standard_ntt(
    uint32_t N, uint64_t Q, int batch_size, int iterations = 100) {

    NTTBenchmarkResult result;
    result.ring_dim = N;
    result.batch_size = batch_size;

    // Create standard engine
    NTTMetalDispatcher engine(N, Q);

    if (!engine.is_gpu_available()) {
        result.correct = false;
        result.forward_time_ms = -1;
        result.inverse_time_ms = -1;
        result.poly_mul_time_ms = -1;
        return result;
    }

    // Generate random test data
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int64_t> dist(0, Q - 1);

    std::vector<int64_t> data(batch_size * N);
    for (auto& v : data) {
        v = dist(rng);
    }

    auto input = mx::array(data.data(), {batch_size, static_cast<int>(N)}, mx::int64);
    mx::eval(input);

    // Warm-up
    for (int i = 0; i < 10; ++i) {
        auto tmp = mx::array(input);
        engine.forward(tmp);
        mx::eval(tmp);
    }

    // Benchmark forward NTT
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto tmp = mx::array(input);
        engine.forward(tmp);
        mx::eval(tmp);
    }
    auto end = std::chrono::high_resolution_clock::now();
    result.forward_time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

    // Benchmark inverse NTT
    auto ntt_data = mx::array(input);
    engine.forward(ntt_data);
    mx::eval(ntt_data);

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto tmp = mx::array(ntt_data);
        engine.inverse(tmp);
        mx::eval(tmp);
    }
    end = std::chrono::high_resolution_clock::now();
    result.inverse_time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

    // Polynomial multiply
    auto a = mx::array(input);
    auto b = mx::array(input);

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        engine.forward(a);
        engine.forward(b);
        auto prod = engine.pointwise_mul(a, b);
        engine.inverse(prod);
        mx::eval(prod);

        // Reset for next iteration
        a = mx::array(input);
        b = mx::array(input);
    }
    end = std::chrono::high_resolution_clock::now();
    result.poly_mul_time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

    // Verify correctness
    auto test = mx::array(input);
    engine.forward(test);
    engine.inverse(test);
    mx::eval(test);

    auto test_ptr = test.data<int64_t>();
    auto orig_ptr = input.data<int64_t>();
    result.correct = true;
    for (int i = 0; i < batch_size * static_cast<int>(N); ++i) {
        if (test_ptr[i] != orig_ptr[i]) {
            result.correct = false;
            break;
        }
    }

    return result;
}

// =============================================================================
// Full Benchmark Suite
// =============================================================================

inline void run_ntt_benchmark_suite() {
    std::cout << "=============================================================\n";
    std::cout << "NTT Batch Optimization Benchmark Suite\n";
    std::cout << "=============================================================\n\n";

    // FHE-friendly primes
    const uint64_t Q_4096 = 0xFFFFFFFF00000001ULL;  // 2^64 - 2^32 + 1
    const uint64_t Q_1024 = 12289;  // NTT-friendly prime for small tests

    struct Config {
        uint32_t N;
        uint64_t Q;
        int batch;
    };

    std::vector<Config> configs = {
        {1024, Q_1024, 1},
        {1024, Q_1024, 8},
        {1024, Q_1024, 64},
        {4096, Q_4096, 1},
        {4096, Q_4096, 8},
        {4096, Q_4096, 64},
    };

    std::cout << "Standard NTT (baseline):\n";
    std::cout << "-------------------------\n";
    std::vector<NTTBenchmarkResult> baseline_results;
    for (const auto& cfg : configs) {
        auto result = benchmark_standard_ntt(cfg.N, cfg.Q, cfg.batch, 50);
        result.print("  Standard");
        baseline_results.push_back(result);
    }

    std::cout << "\nBatch-Vectorized NTT (optimized):\n";
    std::cout << "----------------------------------\n";
    std::vector<NTTBenchmarkResult> optimized_results;
    for (const auto& cfg : configs) {
        auto result = benchmark_batch_vectorized(cfg.N, cfg.Q, cfg.batch, 50);
        result.print("  Optimized");
        optimized_results.push_back(result);
    }

    std::cout << "\nSpeedup (optimized vs baseline):\n";
    std::cout << "---------------------------------\n";
    for (size_t i = 0; i < configs.size(); ++i) {
        double fwd_speedup = baseline_results[i].forward_time_ms / optimized_results[i].forward_time_ms;
        double inv_speedup = baseline_results[i].inverse_time_ms / optimized_results[i].inverse_time_ms;
        double mul_speedup = baseline_results[i].poly_mul_time_ms / optimized_results[i].poly_mul_time_ms;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  N=" << configs[i].N << ", B=" << configs[i].batch
                  << " | fwd=" << fwd_speedup << "x"
                  << " | inv=" << inv_speedup << "x"
                  << " | mul=" << mul_speedup << "x"
                  << std::endl;
    }

    std::cout << "\n=============================================================\n";
}

#endif // WITH_MLX

}  // namespace metal
}  // namespace gpu
}  // namespace lbcrypto

#endif // LBCRYPTO_MATH_HAL_MLX_NTT_BATCH_BENCHMARK_H
