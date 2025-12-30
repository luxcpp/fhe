// =============================================================================
// NTT Correctness Tests
// =============================================================================
//
// Verify:
// 1. INTT(NTT(x)) = x (roundtrip)
// 2. NTT(a * b) = NTT(a) ⊙ NTT(b) (convolution theorem)
// 3. Montgomery arithmetic correctness

#include "ntt.h"
#include <iostream>
#include <vector>
#include <random>
#include <cassert>

using namespace lbcrypto::gpu;

// =============================================================================
// Test Helpers
// =============================================================================

// Generate random polynomial with coefficients in [0, Q)
std::vector<uint64_t> random_poly(uint32_t N, uint64_t Q, std::mt19937_64& rng) {
    std::uniform_int_distribution<uint64_t> dist(0, Q - 1);
    std::vector<uint64_t> poly(N);
    for (uint32_t i = 0; i < N; ++i) {
        poly[i] = dist(rng);
    }
    return poly;
}

// Naive O(N^2) cyclic polynomial multiplication for reference
// This computes a*b mod (X^N - 1), i.e., cyclic convolution
std::vector<uint64_t> naive_poly_mul(const std::vector<uint64_t>& a,
                                      const std::vector<uint64_t>& b,
                                      uint64_t Q) {
    uint32_t N = a.size();
    std::vector<uint64_t> result(N, 0);

    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            uint64_t prod = mulmod(a[i], b[j], Q);
            uint32_t idx = (i + j) % N;  // Cyclic: wrap around
            result[idx] = (result[idx] + prod) % Q;
        }
    }
    return result;
}

// =============================================================================
// Test: Montgomery Arithmetic
// =============================================================================

bool test_montgomery() {
    std::cout << "Test: Montgomery arithmetic... ";

    // Use smaller prime for simpler testing: 65537 (Fermat prime)
    uint64_t Q = 65537ULL;
    uint64_t R = 1ULL << 32;  // R > Q required

    // Compute Montgomery parameters
    // Q_inv: Q * Q_inv ≡ 1 (mod R)
    uint64_t Q_inv = mod_inverse(Q, R);
    // Q': Q * Q' ≡ -1 (mod R), so Q' = R - Q_inv
    uint64_t Q_neg_inv = (R - Q_inv) % R;
    // R^2 mod Q for converting to Montgomery form
    uint64_t R_mod_Q = R % Q;
    uint64_t R_sq_mod_Q = mulmod(R_mod_Q, R_mod_Q, Q);

    // Test values
    uint64_t a = 12345ULL % Q;
    uint64_t b = 54321ULL % Q;

    // Convert to Montgomery form: aR mod Q
    uint64_t a_mont = mulmod(a, R_sq_mod_Q, Q);
    uint64_t b_mont = mulmod(b, R_sq_mod_Q, Q);

    // Montgomery multiply using the efficient REDC formula
    // Computes (a_mont * b_mont) * R^{-1} mod Q
    __uint128_t full_prod = (__uint128_t)a_mont * b_mont;

    // m = (full_prod mod R) * Q' mod R
    uint64_t m = (((uint64_t)full_prod) * Q_neg_inv) & (R - 1);
    // t = (full_prod + m * Q) / R
    __uint128_t tmp = full_prod + (__uint128_t)m * Q;
    uint64_t t = (uint64_t)(tmp >> 32);
    uint64_t ab_mont = (t >= Q) ? (t - Q) : t;

    // ab_mont = ab * R mod Q. To get ab mod Q, multiply by 1 and REDC
    // REDC(ab_mont * 1) = ab_mont * R^{-1} mod Q = ab mod Q
    uint64_t m2 = (ab_mont * Q_neg_inv) & (R - 1);
    __uint128_t tmp2 = (__uint128_t)ab_mont + (__uint128_t)m2 * Q;
    uint64_t t2 = (uint64_t)(tmp2 >> 32);
    uint64_t result = (t2 >= Q) ? (t2 - Q) : t2;

    // Verify against direct computation
    uint64_t expected = mulmod(a, b, Q);

    if (result != expected) {
        std::cout << "FAIL\n";
        std::cout << "  Expected: " << expected << ", Got: " << result << "\n";
        return false;
    }

    std::cout << "PASS\n";
    return true;
}

// =============================================================================
// Test: NTT Parameters
// =============================================================================

bool test_ntt_params() {
    std::cout << "Test: NTT parameter computation... ";

    uint32_t N = 1024;
    // NTT-friendly prime: Q ≡ 1 (mod 2N) required for primitive 2N-th root
    // 268441601 = 131075 * 2048 + 1
    uint64_t Q = 268441601ULL;
    
    try {
        NTTParams params = NTTParams::create(N, Q);
        
        // Verify Q-1 is divisible by 2N
        if ((Q - 1) % (2 * N) != 0) {
            std::cout << "FAIL: Q-1 not divisible by 2N\n";
            return false;
        }
        
        // Verify primitive root exists
        uint64_t omega = find_primitive_root(N, Q);
        
        // omega^{2N} should equal 1
        uint64_t omega_2N = powmod(omega, 2 * N, Q);
        if (omega_2N != 1) {
            std::cout << "FAIL: omega^{2N} != 1\n";
            return false;
        }
        
        // omega^N should equal -1 (mod Q)
        uint64_t omega_N = powmod(omega, N, Q);
        if (omega_N != Q - 1) {
            std::cout << "FAIL: omega^N != -1\n";
            return false;
        }
        
        std::cout << "PASS\n";
        std::cout << "  N = " << N << ", Q = " << Q << "\n";
        std::cout << "  omega = " << omega << "\n";
        std::cout << "  log_N = " << params.log_N << "\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "FAIL: " << e.what() << "\n";
        return false;
    }
}

// =============================================================================
// Test: NTT Roundtrip
// =============================================================================

#ifdef WITH_MLX
bool test_ntt_roundtrip() {
    std::cout << "Test: NTT roundtrip (INTT(NTT(x)) = x)... ";
    
    uint32_t N = 64;  // Small for testing
    uint64_t Q = 65537;  // Fermat prime, 2^16 + 1
    
    // Verify Q-1 divisible by 2N
    if ((Q - 1) % (2 * N) != 0) {
        std::cout << "SKIP: Q-1 not divisible by 2N\n";
        return true;
    }
    
    try {
        NTTEngine engine(N, Q);
        
        std::mt19937_64 rng(42);
        auto poly = random_poly(N, Q, rng);
        
        // Convert to MLX array
        std::vector<int64_t> poly_i64(poly.begin(), poly.end());
        auto data = mx::array(poly_i64.data(), {1, (int)N}, mx::int64);
        mx::eval(data);
        
        // Save original
        auto original = mx::array(data);
        
        // Forward NTT
        engine.forward(data);
        
        // Inverse NTT
        engine.inverse(data);
        
        // Compare
        mx::eval(data);
        auto result_ptr = data.data<int64_t>();
        
        bool pass = true;
        for (uint32_t i = 0; i < N; ++i) {
            uint64_t expected = poly[i];
            uint64_t got = static_cast<uint64_t>(result_ptr[i]) % Q;
            if (expected != got) {
                std::cout << "FAIL at index " << i << "\n";
                std::cout << "  Expected: " << expected << ", Got: " << got << "\n";
                pass = false;
                break;
            }
        }
        
        if (pass) {
            std::cout << "PASS\n";
        }
        return pass;
        
    } catch (const std::exception& e) {
        std::cout << "FAIL: " << e.what() << "\n";
        return false;
    }
}

// =============================================================================
// Test: Polynomial Multiplication
// =============================================================================

bool test_poly_mul() {
    std::cout << "Test: NTT polynomial multiplication... ";
    
    uint32_t N = 64;
    uint64_t Q = 65537;
    
    if ((Q - 1) % (2 * N) != 0) {
        std::cout << "SKIP: Q-1 not divisible by 2N\n";
        return true;
    }
    
    try {
        NTTEngine engine(N, Q);
        
        std::mt19937_64 rng(42);
        auto a = random_poly(N, Q, rng);
        auto b = random_poly(N, Q, rng);
        
        // Reference result
        auto expected = naive_poly_mul(a, b, Q);
        
        // NTT-based multiplication
        std::vector<int64_t> a_i64(a.begin(), a.end());
        std::vector<int64_t> b_i64(b.begin(), b.end());
        
        auto a_arr = mx::array(a_i64.data(), {1, (int)N}, mx::int64);
        auto b_arr = mx::array(b_i64.data(), {1, (int)N}, mx::int64);
        
        auto result = engine.poly_mul(a_arr, b_arr);
        mx::eval(result);
        auto result_ptr = result.data<int64_t>();
        
        bool pass = true;
        for (uint32_t i = 0; i < N; ++i) {
            uint64_t exp = expected[i];
            uint64_t got = static_cast<uint64_t>(result_ptr[i]) % Q;
            if (exp != got) {
                std::cout << "FAIL at index " << i << "\n";
                std::cout << "  Expected: " << exp << ", Got: " << got << "\n";
                pass = false;
                break;
            }
        }
        
        if (pass) {
            std::cout << "PASS\n";
        }
        return pass;
        
    } catch (const std::exception& e) {
        std::cout << "FAIL: " << e.what() << "\n";
        return false;
    }
}
#endif // WITH_MLX

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "=== NTT Correctness Tests ===\n\n";
    
    int passed = 0;
    int total = 0;
    
    ++total; if (test_montgomery()) ++passed;
    ++total; if (test_ntt_params()) ++passed;
    
#ifdef WITH_MLX
    ++total; if (test_ntt_roundtrip()) ++passed;
    ++total; if (test_poly_mul()) ++passed;
#else
    std::cout << "Test: NTT roundtrip... SKIP (MLX not available)\n";
    std::cout << "Test: Polynomial multiplication... SKIP (MLX not available)\n";
#endif
    
    std::cout << "\n=== Results: " << passed << "/" << total << " tests passed ===\n";
    
    return (passed == total) ? 0 : 1;
}
