// =============================================================================
// Barrett Metal Reduction Tests
// =============================================================================
//
// Comprehensive tests for constant-time Barrett reduction implementation.
// Tests correctness, edge cases, NTT integration, and batch operations.
//
// Build: clang++ -std=c++17 -DWITH_MLX -I/path/to/mlx barrett_metal_test.cpp -o test
// Run: ./test

#include "barrett_metal.h"
#include <iostream>
#include <random>
#include <chrono>
#include <cassert>
#include <iomanip>

using namespace lux::gpu::metal;

// =============================================================================
// Test Infrastructure
// =============================================================================

struct TestResult {
    std::string name;
    bool passed;
    std::string message;
};

std::vector<TestResult> results;

#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "Running: " << #name << "... " << std::flush; \
    try { \
        test_##name(); \
        std::cout << "PASSED" << std::endl; \
        results.push_back({#name, true, ""}); \
    } catch (const std::exception& e) { \
        std::cout << "FAILED: " << e.what() << std::endl; \
        results.push_back({#name, false, e.what()}); \
    } \
} while(0)

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        throw std::runtime_error( \
            std::string("Assertion failed: ") + #a + " != " + #b + \
            " (" + std::to_string(a) + " != " + std::to_string(b) + ")"); \
    } \
} while(0)

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        throw std::runtime_error(std::string("Assertion failed: ") + #cond); \
    } \
} while(0)

// =============================================================================
// Test: Barrett Parameter Creation
// =============================================================================

TEST(barrett_params_creation) {
    // Test with common FHE primes
    uint32_t primes[] = {
        998244353,   // 7 * 17 * 2^23 + 1
        469762049,   // Famous NTT prime
        2013265921,  // 15 * 2^27 + 1 (largest 32-bit NTT prime)
        65537,       // Fermat prime F4
    };

    for (uint32_t Q : primes) {
        Barrett32Params p = Barrett32Params::create(Q, 1024);
        ASSERT_EQ(p.Q, Q);
        ASSERT_TRUE(p.mu > 0);

        // Verify mu is approximately 2^64 / Q
        // mu * Q should be close to 2^64
        __uint128_t product = static_cast<__uint128_t>(p.mu) * Q;
        __uint128_t pow2_64 = static_cast<__uint128_t>(1) << 64;
        ASSERT_TRUE(product <= pow2_64);
        ASSERT_TRUE(product > pow2_64 - Q);
    }
}

// =============================================================================
// Test: Barrett Multiply-Reduce Correctness
// =============================================================================

TEST(barrett_mul_correctness) {
    uint32_t Q = 998244353;
    Barrett32Params params = Barrett32Params::create(Q);

    // Test various input combinations
    struct TestCase {
        uint32_t a, b;
    };

    std::vector<TestCase> cases = {
        {0, 0},
        {1, 1},
        {0, 12345},
        {12345, 0},
        {Q - 1, 1},
        {1, Q - 1},
        {Q - 1, Q - 1},
        {Q / 2, 2},
        {12345678, 87654321},
        {999999, 999999},
    };

    for (const auto& tc : cases) {
        uint64_t expected = (static_cast<uint64_t>(tc.a) * tc.b) % Q;
        uint32_t result = barrett_mul_mod_cpu(tc.a, tc.b, Q, params.mu);
        ASSERT_EQ(result, static_cast<uint32_t>(expected));
    }
}

// =============================================================================
// Test: Random Inputs
// =============================================================================

TEST(barrett_mul_random) {
    uint32_t Q = 998244353;
    Barrett32Params params = Barrett32Params::create(Q);

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, Q - 1);

    for (int i = 0; i < 10000; ++i) {
        uint32_t a = dist(rng);
        uint32_t b = dist(rng);

        uint64_t expected = (static_cast<uint64_t>(a) * b) % Q;
        uint32_t result = barrett_mul_mod_cpu(a, b, Q, params.mu);

        ASSERT_EQ(result, static_cast<uint32_t>(expected));
    }
}

// =============================================================================
// Test: Modular Addition
// =============================================================================

TEST(mod_add_correctness) {
    uint32_t Q = 998244353;

    struct TestCase {
        uint32_t a, b, expected;
    };

    std::vector<TestCase> cases = {
        {0, 0, 0},
        {1, 1, 2},
        {Q - 1, 1, 0},
        {Q - 1, Q - 1, Q - 2},
        {Q / 2, Q / 2, Q - 1},  // Q is odd
        {100, 200, 300},
        {Q - 100, 200, 100},
    };

    for (const auto& tc : cases) {
        uint32_t result = mod_add_cpu(tc.a, tc.b, Q);
        ASSERT_EQ(result, tc.expected);
    }
}

// =============================================================================
// Test: Modular Subtraction
// =============================================================================

TEST(mod_sub_correctness) {
    uint32_t Q = 998244353;

    struct TestCase {
        uint32_t a, b, expected;
    };

    std::vector<TestCase> cases = {
        {0, 0, 0},
        {1, 0, 1},
        {0, 1, Q - 1},
        {100, 50, 50},
        {50, 100, Q - 50},
        {Q - 1, Q - 1, 0},
        {0, Q - 1, 1},
    };

    for (const auto& tc : cases) {
        uint32_t result = mod_sub_cpu(tc.a, tc.b, Q);
        ASSERT_EQ(result, tc.expected);
    }
}

// =============================================================================
// Test: N_inv Computation
// =============================================================================

TEST(n_inv_correctness) {
    uint32_t Q = 998244353;
    uint32_t N = 1024;

    Barrett32Params params = Barrett32Params::create(Q, N);

    // Verify: N * N_inv == 1 (mod Q)
    uint64_t product = (static_cast<uint64_t>(N) * params.N_inv) % Q;
    ASSERT_EQ(product, 1ULL);
}

// =============================================================================
// Test: Different Primes
// =============================================================================

TEST(barrett_different_primes) {
    // Test with various NTT-friendly primes
    uint32_t primes[] = {
        65537,       // 2^16 + 1
        786433,      // 3 * 2^18 + 1
        5767169,     // 11 * 2^19 + 1
        469762049,   // 7 * 2^26 + 1
        998244353,   // 119 * 2^23 + 1
        1004535809,  // 479 * 2^21 + 1
        2013265921,  // 15 * 2^27 + 1
    };

    std::mt19937_64 rng(42);

    for (uint32_t Q : primes) {
        Barrett32Params params = Barrett32Params::create(Q);
        std::uniform_int_distribution<uint32_t> dist(0, Q - 1);

        for (int i = 0; i < 100; ++i) {
            uint32_t a = dist(rng);
            uint32_t b = dist(rng);

            uint64_t expected = (static_cast<uint64_t>(a) * b) % Q;
            uint32_t result = barrett_mul_mod_cpu(a, b, Q, params.mu);

            ASSERT_EQ(result, static_cast<uint32_t>(expected));
        }
    }
}

// =============================================================================
// Test: Built-in Correctness Test
// =============================================================================

TEST(builtin_correctness) {
    ASSERT_TRUE(test_barrett32_correctness());
}

// =============================================================================
// Test: Metal Shader Source Generation
// =============================================================================

TEST(metal_shader_source) {
    const char* source = get_barrett_metal_source();
    ASSERT_TRUE(source != nullptr);
    ASSERT_TRUE(strlen(source) > 1000);  // Sanity check

    // Check for key functions
    std::string src(source);
    ASSERT_TRUE(src.find("barrett_mul_mod") != std::string::npos);
    ASSERT_TRUE(src.find("mod_add") != std::string::npos);
    ASSERT_TRUE(src.find("mod_sub") != std::string::npos);
    ASSERT_TRUE(src.find("ntt_butterfly_ct") != std::string::npos);
    ASSERT_TRUE(src.find("ntt_butterfly_gs") != std::string::npos);
    ASSERT_TRUE(src.find("constant-time") != std::string::npos ||
                src.find("Constant-Time") != std::string::npos);
}

#ifdef WITH_MLX

// =============================================================================
// Test: MLX Barrett Dispatcher
// =============================================================================

TEST(mlx_dispatcher_creation) {
    uint32_t Q = 998244353;
    uint32_t N = 1024;

    Barrett32Dispatcher dispatcher(Q, N);

    ASSERT_EQ(dispatcher.modulus(), Q);
    // Note: is_available() depends on Metal hardware
}

// =============================================================================
// Test: MLX NTT Round-Trip
// =============================================================================

TEST(mlx_ntt_roundtrip) {
    uint32_t Q = 998244353;
    uint32_t N = 16;

    Barrett32Dispatcher dispatcher(Q, N);

    std::vector<int32_t> original(N);
    for (uint32_t i = 0; i < N; ++i) {
        original[i] = static_cast<int32_t>(i * 12345 % Q);
    }

    auto data = mx::array(original.data(), {static_cast<int>(N)}, mx::int32);
    mx::eval(data);

    dispatcher.ntt_forward(data);
    dispatcher.ntt_inverse(data);

    mx::eval(data);
    auto ptr = data.data<int32_t>();

    for (uint32_t i = 0; i < N; ++i) {
        ASSERT_EQ(ptr[i], original[i]);
    }
}

// =============================================================================
// Test: MLX Polynomial Multiplication
// =============================================================================

TEST(mlx_poly_mul) {
    uint32_t Q = 998244353;
    uint32_t N = 16;

    Barrett32Dispatcher dispatcher(Q, N);

    // (1 + x) * (1 + x) = 1 + 2x + x^2
    std::vector<int32_t> a(N, 0), b(N, 0);
    a[0] = 1; a[1] = 1;
    b[0] = 1; b[1] = 1;

    auto a_arr = mx::array(a.data(), {static_cast<int>(N)}, mx::int32);
    auto b_arr = mx::array(b.data(), {static_cast<int>(N)}, mx::int32);

    auto result = dispatcher.poly_mul(a_arr, b_arr);
    mx::eval(result);

    auto ptr = result.data<int32_t>();

    ASSERT_EQ(ptr[0], 1);
    ASSERT_EQ(ptr[1], 2);
    ASSERT_EQ(ptr[2], 1);

    for (uint32_t i = 3; i < N; ++i) {
        ASSERT_EQ(ptr[i], 0);
    }
}

// =============================================================================
// Test: MLX Batch NTT
// =============================================================================

TEST(mlx_batch_ntt) {
    uint32_t Q = 998244353;
    uint32_t N = 16;
    int batch = 4;

    Barrett32Dispatcher dispatcher(Q, N);

    std::vector<int32_t> data(batch * N);
    for (int b = 0; b < batch; ++b) {
        for (uint32_t i = 0; i < N; ++i) {
            data[b * N + i] = static_cast<int32_t>((b * 1000 + i) % Q);
        }
    }

    auto original = data;
    auto arr = mx::array(data.data(), {batch, static_cast<int>(N)}, mx::int32);

    dispatcher.ntt_forward(arr);
    dispatcher.ntt_inverse(arr);

    mx::eval(arr);
    auto ptr = arr.data<int32_t>();

    for (int i = 0; i < batch * static_cast<int>(N); ++i) {
        ASSERT_EQ(ptr[i], original[i]);
    }
}

// =============================================================================
// Test: MLX Built-in NTT Test
// =============================================================================

TEST(mlx_builtin_ntt) {
    ASSERT_TRUE(test_barrett32_ntt());
}

// =============================================================================
// Test: Modular Operations via MLX
// =============================================================================

TEST(mlx_mod_ops) {
    uint32_t Q = 998244353;
    Barrett32Dispatcher dispatcher(Q);

    std::vector<int32_t> a_data = {100, 200, Q - 50, Q - 1};
    std::vector<int32_t> b_data = {50, Q - 100, 100, 1};

    auto a = mx::array(a_data.data(), {4}, mx::int32);
    auto b = mx::array(b_data.data(), {4}, mx::int32);

    // Test add
    auto sum = dispatcher.add_mod(a, b);
    mx::eval(sum);
    auto sum_ptr = sum.data<int32_t>();
    ASSERT_EQ(sum_ptr[0], 150);
    ASSERT_EQ(sum_ptr[1], 100);  // 200 + (Q-100) = Q+100 mod Q = 100
    ASSERT_EQ(sum_ptr[2], 50);   // (Q-50) + 100 = Q+50 mod Q = 50
    ASSERT_EQ(sum_ptr[3], 0);    // (Q-1) + 1 = Q mod Q = 0

    // Test sub
    auto diff = dispatcher.sub_mod(a, b);
    mx::eval(diff);
    auto diff_ptr = diff.data<int32_t>();
    ASSERT_EQ(diff_ptr[0], 50);          // 100 - 50 = 50
    ASSERT_EQ(diff_ptr[1], 300);         // 200 - (Q-100) = 200+100 = 300
    ASSERT_EQ(diff_ptr[2], (int32_t)(Q - 150)); // (Q-50) - 100 = Q - 150
    ASSERT_EQ(diff_ptr[3], (int32_t)(Q - 2));   // (Q-1) - 1 = Q - 2
}

#endif // WITH_MLX

// =============================================================================
// Performance Benchmark
// =============================================================================

void benchmark_barrett() {
    std::cout << "\n=== Performance Benchmark ===" << std::endl;

    uint32_t Q = 998244353;
    Barrett32Params params = Barrett32Params::create(Q);

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, Q - 1);

    const int N = 1000000;
    std::vector<uint32_t> a(N), b(N), c(N);

    for (int i = 0; i < N; ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }

    // Warm up
    for (int i = 0; i < N; ++i) {
        c[i] = barrett_mul_mod_cpu(a[i], b[i], Q, params.mu);
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < 10; ++iter) {
        for (int i = 0; i < N; ++i) {
            c[i] = barrett_mul_mod_cpu(a[i], b[i], Q, params.mu);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double ops_per_sec = (10.0 * N) / (duration.count() / 1e6);
    std::cout << "Barrett mul_mod: " << std::fixed << std::setprecision(2)
              << ops_per_sec / 1e6 << " M ops/sec" << std::endl;

    // Prevent optimization
    volatile uint32_t sink = c[0];
    (void)sink;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "=== Barrett Metal Reduction Tests ===" << std::endl;
    std::cout << std::endl;

    // CPU tests
    RUN_TEST(barrett_params_creation);
    RUN_TEST(barrett_mul_correctness);
    RUN_TEST(barrett_mul_random);
    RUN_TEST(mod_add_correctness);
    RUN_TEST(mod_sub_correctness);
    RUN_TEST(n_inv_correctness);
    RUN_TEST(barrett_different_primes);
    RUN_TEST(builtin_correctness);
    RUN_TEST(metal_shader_source);

#ifdef WITH_MLX
    std::cout << "\n--- MLX/GPU Tests ---" << std::endl;
    RUN_TEST(mlx_dispatcher_creation);
    RUN_TEST(mlx_ntt_roundtrip);
    RUN_TEST(mlx_poly_mul);
    RUN_TEST(mlx_batch_ntt);
    RUN_TEST(mlx_builtin_ntt);
    RUN_TEST(mlx_mod_ops);
#else
    std::cout << "\n[SKIP] MLX tests (WITH_MLX not defined)" << std::endl;
#endif

    // Performance benchmark
    benchmark_barrett();

    // Summary
    std::cout << "\n=== Test Summary ===" << std::endl;
    int passed = 0, failed = 0;
    for (const auto& r : results) {
        if (r.passed) {
            passed++;
        } else {
            failed++;
            std::cout << "  FAILED: " << r.name << ": " << r.message << std::endl;
        }
    }
    std::cout << "Passed: " << passed << "/" << (passed + failed) << std::endl;

    return failed == 0 ? 0 : 1;
}
