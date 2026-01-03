//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2024, Lux Industries Inc
//
// Unit tests for MLX GPU Backend
//==================================================================================

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <chrono>

#include "math/hal/mlx/mlx_backend.h"

using namespace lux::mlx_backend;

class MLXBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if MLX is available
        mlx_available_ = IsMLXAvailable();
        if (!mlx_available_) {
            GTEST_SKIP() << "MLX not available, skipping tests";
        }
    }
    
    bool mlx_available_ = false;
};

#ifdef WITH_MLX

TEST_F(MLXBackendTest, CheckAvailability) {
    if (!mlx_available_) GTEST_SKIP();
    
    EXPECT_TRUE(IsMLXAvailable());
    std::cout << "MLX Device: " << GetDeviceName() << std::endl;
}

TEST_F(MLXBackendTest, NTTRoundtrip) {
    if (!mlx_available_) GTEST_SKIP();
    
    // Test parameters
    const uint64_t n = 512;  // Ring dimension
    const uint64_t q = 12289;  // Prime modulus (12289 = 3 * 4096 + 1)
    
    MLXNTT ntt(n, q);
    
    // Create random polynomial
    std::vector<uint64_t> input(n);
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist(0, q - 1);
    
    for (size_t i = 0; i < n; ++i) {
        input[i] = dist(gen);
    }
    
    // Forward NTT
    std::vector<uint64_t> ntt_output;
    ntt.ForwardTransform(input, ntt_output);
    
    EXPECT_EQ(ntt_output.size(), n);
    
    // Inverse NTT
    std::vector<uint64_t> recovered;
    ntt.InverseTransform(ntt_output, recovered);
    
    EXPECT_EQ(recovered.size(), n);
    
    // Check roundtrip
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(recovered[i], input[i]) << "Mismatch at index " << i;
    }
}

TEST_F(MLXBackendTest, NTTMultiplication) {
    if (!mlx_available_) GTEST_SKIP();
    
    // Test polynomial multiplication via NTT
    const uint64_t n = 256;
    const uint64_t q = 12289;
    
    MLXNTT ntt(n, q);
    
    // Simple test: multiply (1 + X) * (1 + X) = 1 + 2X + X^2
    std::vector<uint64_t> a(n, 0);
    std::vector<uint64_t> b(n, 0);
    a[0] = 1;  // Constant term
    a[1] = 1;  // X term
    b[0] = 1;
    b[1] = 1;
    
    // NTT forward
    std::vector<uint64_t> a_ntt, b_ntt;
    ntt.ForwardTransform(a, a_ntt);
    ntt.ForwardTransform(b, b_ntt);
    
    // Element-wise multiply
    std::vector<uint64_t> c_ntt;
    ntt.ElementwiseMultMod(a_ntt, b_ntt, c_ntt);
    
    // NTT inverse
    std::vector<uint64_t> c;
    ntt.InverseTransform(c_ntt, c);
    
    // Expected: 1 + 2X + X^2
    EXPECT_EQ(c[0], 1);
    EXPECT_EQ(c[1], 2);
    EXPECT_EQ(c[2], 1);
    for (size_t i = 3; i < n; ++i) {
        EXPECT_EQ(c[i], 0);
    }
}

TEST_F(MLXBackendTest, PolyOpsAddSub) {
    if (!mlx_available_) GTEST_SKIP();
    
    const uint64_t n = 128;
    const uint64_t q = 12289;
    
    MLXPolyOps ops(n, q);
    
    std::vector<uint64_t> a(n), b(n), result(n);
    for (size_t i = 0; i < n; ++i) {
        a[i] = i % q;
        b[i] = (2 * i) % q;
    }
    
    // Test addition
    ops.PolyAdd(a, b, result);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(result[i], (a[i] + b[i]) % q);
    }
    
    // Test subtraction
    ops.PolySub(a, b, result);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(result[i], (a[i] + q - b[i]) % q);
    }
    
    // Test negation
    ops.PolyNeg(a, result);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(result[i], (q - a[i]) % q);
    }
}

TEST_F(MLXBackendTest, NTTBenchmark) {
    if (!mlx_available_) GTEST_SKIP();
    
    // Benchmark NTT performance
    const uint64_t n = 1024;
    const uint64_t q = 12289;
    const int iterations = 100;
    
    MLXNTT ntt(n, q);
    
    // Create random polynomial
    std::vector<uint64_t> input(n);
    std::random_device rd;
    std::mt19937_64 gen(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<uint64_t> dist(0, q - 1);
    
    for (size_t i = 0; i < n; ++i) {
        input[i] = dist(gen);
    }
    
    std::vector<uint64_t> output;
    
    // Warmup
    ntt.ForwardTransform(input, output);
    
    // Benchmark forward NTT
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        ntt.ForwardTransform(input, output);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_us = duration.count() / static_cast<double>(iterations);
    
    std::cout << "NTT Benchmark (n=" << n << "):" << std::endl;
    std::cout << "  Forward NTT: " << avg_us << " µs per transform" << std::endl;
    std::cout << "  Throughput: " << (1000000.0 / avg_us) << " transforms/sec" << std::endl;
    std::cout << "  GPU enabled: " << (ntt.IsGPUEnabled() ? "yes" : "no") << std::endl;
    std::cout << "  GPU memory: " << GetGPUMemoryUsage() / 1024 << " KB" << std::endl;
}

TEST_F(MLXBackendTest, BatchNTTPerformance) {
    if (!mlx_available_) GTEST_SKIP();
    
    // Test batch NTT performance
    const uint64_t n = 512;
    const uint64_t q = 12289;
    const size_t batch_size = 32;
    
    MLXNTT ntt(n, q);
    
    // Create batch of random polynomials
    std::vector<std::vector<uint64_t>> inputs(batch_size);
    std::random_device rd;
    std::mt19937_64 gen(42);
    std::uniform_int_distribution<uint64_t> dist(0, q - 1);
    
    for (size_t b = 0; b < batch_size; ++b) {
        inputs[b].resize(n);
        for (size_t i = 0; i < n; ++i) {
            inputs[b][i] = dist(gen);
        }
    }
    
    std::vector<std::vector<uint64_t>> outputs;
    
    // Benchmark batch NTT
    auto start = std::chrono::high_resolution_clock::now();
    ntt.ForwardTransformBatch(inputs, outputs);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Batch NTT (batch_size=" << batch_size << ", n=" << n << "):" << std::endl;
    std::cout << "  Total time: " << duration.count() << " µs" << std::endl;
    std::cout << "  Per polynomial: " << duration.count() / batch_size << " µs" << std::endl;
}

#else

TEST(MLXBackendTest, NotCompiled) {
    EXPECT_FALSE(IsMLXAvailable());
    std::cout << "MLX not compiled in (use -DWITH_MLX=ON to enable)" << std::endl;
}

#endif // WITH_MLX
