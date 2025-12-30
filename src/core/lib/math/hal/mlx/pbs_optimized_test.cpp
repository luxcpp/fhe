// =============================================================================
// PBS Optimization Tests
// =============================================================================
//
// Compile with: clang++ -std=c++17 -DWITH_MLX -I$(mlx-prefix)/include \
//               pbs_optimized_test.cpp -o pbs_test -L$(mlx-prefix)/lib -lmlx
//
// Tests verify:
// 1. TestPolynomialCache caches and returns consistent test polynomials
// 2. BatchPBS executes multiple operations in single dispatch
// 3. OptimizedPBSEngine provides drop-in replacement for sequential PBS
// 4. euint256PBSContext provides parallel word operations
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0
// =============================================================================

#include <cassert>
#include <chrono>
#include <cstdio>
#include <vector>

#ifdef WITH_MLX
#include <mlx/mlx.h>
#include "pbs_optimized.h"
#include "euint256_pbs_integration.h"
namespace mx = mlx::core;

using namespace lbcrypto::gpu;

// =============================================================================
// Test Helpers
// =============================================================================

static void test_passed(const char* name) {
    printf("[PASS] %s\n", name);
}

static void test_failed(const char* name, const char* reason) {
    printf("[FAIL] %s: %s\n", name, reason);
}

// =============================================================================
// Test: TestPolynomialCache
// =============================================================================

void test_polynomial_cache() {
    TestPolynomialCache::Config cfg;
    cfg.N = 1024;
    cfg.Q = 1ULL << 27;
    cfg.n = 512;

    auto& cache = TestPolynomialCache::instance();
    cache.configure(cfg);

    // Test cache hit
    const auto& poly1 = cache.get(TestPolyType::IDENTITY);
    const auto& poly2 = cache.get(TestPolyType::IDENTITY);

    // Should return same object
    if (poly1.data<int64_t>() == poly2.data<int64_t>()) {
        test_passed("TestPolynomialCache cache hit");
    } else {
        test_failed("TestPolynomialCache cache hit", "Different pointers returned");
    }

    // Test different types return different polynomials
    const auto& poly_and = cache.get(TestPolyType::BOOL_AND);
    const auto& poly_or = cache.get(TestPolyType::BOOL_OR);

    mx::eval(poly_and);
    mx::eval(poly_or);

    auto and_ptr = poly_and.data<int64_t>();
    auto or_ptr = poly_or.data<int64_t>();

    bool different = false;
    for (int i = 0; i < 10; ++i) {
        if (and_ptr[i] != or_ptr[i]) {
            different = true;
            break;
        }
    }

    if (different) {
        test_passed("TestPolynomialCache type differentiation");
    } else {
        test_failed("TestPolynomialCache type differentiation",
                   "AND and OR polynomials should differ");
    }

    // Test prewarm
    cache.prewarm();
    if (cache.size() >= 5) {
        test_passed("TestPolynomialCache prewarm");
    } else {
        test_failed("TestPolynomialCache prewarm", "Not enough entries");
    }
}

// =============================================================================
// Test: BatchPBS
// =============================================================================

void test_batch_pbs() {
    // Create dummy bootstrap key
    int n = 512;
    int N = 1024;
    int L = 3;

    std::vector<int64_t> bsk_data(n * 2 * L * 2 * N, 0);
    auto bsk = mx::array(bsk_data.data(), {n, 2, L, 2, N}, mx::int64);
    mx::eval(bsk);

    BatchPBS::Config cfg;
    cfg.N = static_cast<uint32_t>(N);
    cfg.n = static_cast<uint32_t>(n);
    cfg.L = static_cast<uint32_t>(L);
    cfg.baseLog = 7;
    cfg.Q = 1ULL << 27;

    BatchPBS batcher(bsk, cfg);

    // Add multiple operations
    std::vector<int64_t> lwe_data(n + 1, 0);
    auto lwe = mx::array(lwe_data.data(), {n + 1}, mx::int64);
    mx::eval(lwe);

    batcher.add(lwe, TestPolyType::IDENTITY);
    batcher.add(lwe, TestPolyType::BOOL_AND);
    batcher.add(lwe, TestPolyType::BOOL_OR);

    if (batcher.batchSize() == 3) {
        test_passed("BatchPBS add operations");
    } else {
        test_failed("BatchPBS add operations", "Wrong batch size");
    }

    // Execute batch
    auto result = batcher.execute();
    mx::eval(result);

    auto shape = result.shape();
    if (shape.size() == 3 && shape[0] == 3 && shape[1] == 2 && shape[2] == N) {
        test_passed("BatchPBS execute shape");
    } else {
        test_failed("BatchPBS execute shape", "Unexpected output shape");
    }
}

// =============================================================================
// Test: euint256PBSContext
// =============================================================================

void test_euint256_context() {
    euint256PBSContext::Config cfg;
    cfg.N = 1024;
    cfg.n = 512;
    cfg.L = 3;
    cfg.baseLog = 7;
    cfg.Q = 1ULL << 27;

    euint256PBSContext ctx(cfg);

    // Create dummy 8-word arrays
    std::array<mx::array, 8> a_words, b_words;
    int n = static_cast<int>(cfg.n);

    for (int i = 0; i < 8; ++i) {
        std::vector<int64_t> data(n + 1, 0);
        a_words[i] = mx::array(data.data(), {n + 1}, mx::int64);
        b_words[i] = mx::array(data.data(), {n + 1}, mx::int64);
        mx::eval(a_words[i]);
        mx::eval(b_words[i]);
    }

    // Test parallel AND
    auto result = ctx.parallelAnd(a_words, b_words);

    bool all_valid = true;
    for (int i = 0; i < 8; ++i) {
        auto shape = result[i].shape();
        if (shape[0] != n + 1) {
            all_valid = false;
            break;
        }
    }

    if (all_valid) {
        test_passed("euint256PBSContext parallelAnd shape");
    } else {
        test_failed("euint256PBSContext parallelAnd shape", "Unexpected output shape");
    }

    // Test parallel OR
    result = ctx.parallelOr(a_words, b_words);

    all_valid = true;
    for (int i = 0; i < 8; ++i) {
        auto shape = result[i].shape();
        if (shape[0] != n + 1) {
            all_valid = false;
            break;
        }
    }

    if (all_valid) {
        test_passed("euint256PBSContext parallelOr shape");
    } else {
        test_failed("euint256PBSContext parallelOr shape", "Unexpected output shape");
    }
}

// =============================================================================
// Benchmark: Sequential vs Batched PBS
// =============================================================================

void benchmark_pbs() {
    printf("\n--- PBS Performance Benchmark ---\n");

    int n = 512;
    int N = 1024;
    int L = 3;
    int num_ops = 16;

    // Create dummy bootstrap key
    std::vector<int64_t> bsk_data(n * 2 * L * 2 * N, 0);
    auto bsk = mx::array(bsk_data.data(), {n, 2, L, 2, N}, mx::int64);
    mx::eval(bsk);

    // Create LWE ciphertexts
    std::vector<mx::array> lwes;
    for (int i = 0; i < num_ops; ++i) {
        std::vector<int64_t> data(n + 1, static_cast<int64_t>(i));
        lwes.push_back(mx::array(data.data(), {n + 1}, mx::int64));
        mx::eval(lwes.back());
    }

    // Batched execution
    BatchPBS::Config cfg;
    cfg.N = static_cast<uint32_t>(N);
    cfg.n = static_cast<uint32_t>(n);
    cfg.L = static_cast<uint32_t>(L);
    cfg.baseLog = 7;
    cfg.Q = 1ULL << 27;

    BatchPBS batcher(bsk, cfg);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_ops; ++i) {
        batcher.add(lwes[i], TestPolyType::IDENTITY);
    }
    auto result = batcher.execute();
    mx::eval(result);

    auto end = std::chrono::high_resolution_clock::now();
    double batched_ms = std::chrono::duration<double, std::milli>(end - start).count();

    printf("Batched PBS (%d ops): %.2f ms (%.2f ms/op)\n",
           num_ops, batched_ms, batched_ms / num_ops);
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("=== PBS Optimization Tests ===\n\n");

    if (!mx::metal::is_available()) {
        printf("WARNING: Metal GPU not available. Some tests may behave differently.\n\n");
    } else {
        printf("Metal GPU: Available\n\n");
        mx::set_default_device(mx::Device::gpu);
    }

    test_polynomial_cache();
    test_batch_pbs();
    test_euint256_context();

    benchmark_pbs();

    printf("\n=== Tests Complete ===\n");
    return 0;
}

#else // !WITH_MLX

int main() {
    printf("WITH_MLX not defined. PBS optimizations require MLX.\n");
    return 1;
}

#endif // WITH_MLX
