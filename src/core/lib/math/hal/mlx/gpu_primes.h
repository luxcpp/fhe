// =============================================================================
// GPU-Native 32-bit Prime Set for Metal NTT
// =============================================================================
//
// Fixed prime set optimized for GPU computation:
// - 30-32 bit primes: 32x32=64 fits in int64 without overflow
// - NTT-friendly: Q = 1 (mod 2N) for N up to 8192
// - Barrett-ready: precomputed mu = floor(2^64 / Q)
// - Stage-aligned: precomputed primitive roots and twiddles
//
// Design rationale:
// - GPU SIMD units (Metal SIMD-groups) work best with uniform 32-bit ops
// - Avoids 128-bit intermediate products that require emulation
// - Better register utilization than 64-bit primes
// - Sufficient security for FHE when using RNS with multiple primes
//
// Prime selection criteria:
// - Q in range [2^30, 2^32) for maximum precision
// - Q - 1 divisible by 2*8192 = 16384 (supports N up to 8192)
// - Q is prime (verified via Miller-Rabin)
// - Primitive 2N-th root exists (guaranteed by Q = 1 mod 2N)
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

#ifndef LUX_FHE_MATH_HAL_MLX_GPU_PRIMES_H
#define LUX_FHE_MATH_HAL_MLX_GPU_PRIMES_H

#include <cstdint>
#include <vector>
#include <array>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>

namespace lux {
namespace gpu {

// =============================================================================
// GPU32Prime - Single prime with precomputed constants
// =============================================================================

struct GPU32Prime {
    uint32_t q;           // Prime modulus (30-32 bits)
    uint32_t psi;         // Primitive 2N-th root of unity (for negacyclic NTT)
    uint32_t psi_inv;     // psi^{-1} mod q
    uint32_t omega;       // Primitive N-th root = psi^2 (for standard NTT stages)
    uint32_t omega_inv;   // omega^{-1} mod q
    uint32_t n_inv;       // N^{-1} mod q (for inverse NTT scaling)
    uint64_t barrett_mu;  // floor(2^64 / q) for Barrett reduction
    uint32_t N;           // Ring dimension this prime supports
    uint32_t log_N;       // log2(N)

    // Verify prime properties at construction
    bool verify() const;

    // Compute Barrett reduction: floor(a * mu >> 64) approximates floor(a / q)
    // For a < q^2 < 2^64, result is in {0, 1, 2} * q of true value
    inline uint32_t barrett_reduce(uint64_t a) const {
        uint64_t q_approx = static_cast<uint64_t>(((__uint128_t)a * barrett_mu) >> 64);
        uint64_t r = a - q_approx * q;
        // At most 2 conditional subtractions
        if (r >= q) r -= q;
        if (r >= q) r -= q;
        return static_cast<uint32_t>(r);
    }

    // Modular multiplication using Barrett
    inline uint32_t mulmod(uint32_t a, uint32_t b) const {
        return barrett_reduce(static_cast<uint64_t>(a) * b);
    }

    // Modular addition
    inline uint32_t addmod(uint32_t a, uint32_t b) const {
        uint32_t sum = a + b;
        return (sum >= q) ? sum - q : sum;
    }

    // Modular subtraction
    inline uint32_t submod(uint32_t a, uint32_t b) const {
        return (a >= b) ? a - b : a + q - b;
    }

    // Create prime with all precomputed constants
    static GPU32Prime create(uint32_t q, uint32_t N);
};

// =============================================================================
// Modular arithmetic helpers (compile-time and runtime)
// =============================================================================

namespace detail {

// Extended GCD for modular inverse
inline void extended_gcd_32(uint32_t a, uint32_t b, int32_t& g, int32_t& x, int32_t& y) {
    if (b == 0) {
        g = static_cast<int32_t>(a);
        x = 1;
        y = 0;
        return;
    }
    int32_t g1, x1, y1;
    extended_gcd_32(b, a % b, g1, x1, y1);
    g = g1;
    x = y1;
    y = x1 - static_cast<int32_t>(a / b) * y1;
}

inline uint32_t mod_inverse_32(uint32_t a, uint32_t m) {
    int32_t g, x, y;
    extended_gcd_32(a, m, g, x, y);
    if (g != 1) {
        throw std::runtime_error("mod_inverse_32: inverse does not exist");
    }
    return static_cast<uint32_t>((x % static_cast<int32_t>(m) + static_cast<int32_t>(m)) % static_cast<int32_t>(m));
}

inline uint32_t powmod_32(uint32_t base, uint32_t exp, uint32_t m) {
    uint64_t result = 1;
    uint64_t b = base % m;
    while (exp > 0) {
        if (exp & 1) result = result * b % m;
        b = b * b % m;
        exp >>= 1;
    }
    return static_cast<uint32_t>(result);
}

// Miller-Rabin primality test for 32-bit integers
// Deterministic for n < 2^32 with witnesses {2, 7, 61}
inline bool is_prime_32(uint32_t n) {
    if (n < 2) return false;
    if (n == 2 || n == 7 || n == 61) return true;
    if (n % 2 == 0) return false;

    // Write n-1 as 2^r * d
    uint32_t d = n - 1;
    uint32_t r = 0;
    while ((d & 1) == 0) {
        d >>= 1;
        ++r;
    }

    // Witnesses sufficient for n < 2^32
    const uint32_t witnesses[] = {2, 7, 61};

    for (uint32_t a : witnesses) {
        if (a >= n) continue;

        uint64_t x = powmod_32(a, d, n);
        if (x == 1 || x == n - 1) continue;

        bool composite = true;
        for (uint32_t i = 0; i < r - 1; ++i) {
            x = x * x % n;
            if (x == n - 1) {
                composite = false;
                break;
            }
        }
        if (composite) return false;
    }
    return true;
}

// Find primitive 2N-th root of unity mod Q
// Requires Q = 1 (mod 2N)
inline uint32_t find_primitive_root_32(uint32_t N, uint32_t Q) {
    uint32_t order = Q - 1;
    if (order % (2 * N) != 0) {
        throw std::runtime_error("Q - 1 must be divisible by 2N");
    }

    // Find a generator g of (Z/QZ)* and compute g^((Q-1)/(2N))
    for (uint32_t g = 2; g < Q; ++g) {
        // g is a generator if g^((Q-1)/2) = -1 (mod Q)
        if (powmod_32(g, order / 2, Q) != 1) {
            uint32_t omega = powmod_32(g, order / (2 * N), Q);
            // Verify omega has exact order 2N
            if (powmod_32(omega, N, Q) == Q - 1 && powmod_32(omega, 2 * N, Q) == 1) {
                return omega;
            }
        }
    }
    throw std::runtime_error("No primitive 2N-th root found");
}

}  // namespace detail

// =============================================================================
// GPU32Prime Implementation
// =============================================================================

inline GPU32Prime GPU32Prime::create(uint32_t q, uint32_t N) {
    GPU32Prime p;
    p.q = q;
    p.N = N;
    p.log_N = 0;
    while ((1u << p.log_N) < N) ++p.log_N;

    if ((1u << p.log_N) != N) {
        throw std::runtime_error("N must be power of 2");
    }

    // Verify Q = 1 (mod 2N)
    if ((q - 1) % (2 * N) != 0) {
        throw std::runtime_error("Q - 1 must be divisible by 2N");
    }

    // Barrett constant
    p.barrett_mu = static_cast<uint64_t>(((__uint128_t)1 << 64) / q);

    // Find psi: primitive 2N-th root of unity (for negacyclic convolution)
    // psi^N = -1 (mod q), psi^(2N) = 1 (mod q)
    p.psi = detail::find_primitive_root_32(N, q);
    p.psi_inv = detail::mod_inverse_32(p.psi, q);

    // omega = psi^2: primitive N-th root of unity (for NTT stages)
    // omega^(N/2) = -1 (mod q), omega^N = 1 (mod q)
    p.omega = static_cast<uint32_t>((static_cast<uint64_t>(p.psi) * p.psi) % q);
    p.omega_inv = detail::mod_inverse_32(p.omega, q);

    // N inverse for scaling in inverse NTT
    p.n_inv = detail::mod_inverse_32(N, q);

    return p;
}

inline bool GPU32Prime::verify() const {
    // Check prime
    if (!detail::is_prime_32(q)) return false;

    // Check psi order (primitive 2N-th root)
    if (detail::powmod_32(psi, N, q) != q - 1) return false;
    if (detail::powmod_32(psi, 2 * N, q) != 1) return false;

    // Check omega = psi^2 order (primitive N-th root)
    if (detail::powmod_32(omega, N / 2, q) != q - 1) return false;
    if (detail::powmod_32(omega, N, q) != 1) return false;

    // Check inverses
    uint64_t prod = static_cast<uint64_t>(psi) * psi_inv % q;
    if (prod != 1) return false;

    prod = static_cast<uint64_t>(omega) * omega_inv % q;
    if (prod != 1) return false;

    prod = static_cast<uint64_t>(N) * n_inv % q;
    if (prod != 1) return false;

    return true;
}

// =============================================================================
// Precomputed Twiddle Table
// =============================================================================

struct GPU32TwiddleTable {
    // Negacyclic twist factors: psi_powers[i] = psi^i for i in [0, N)
    std::vector<uint32_t> psi_powers;       // For forward twist
    std::vector<uint32_t> psi_inv_powers;   // For inverse untwist

    // NTT butterfly twiddles using omega (N-th root)
    std::vector<uint32_t> forward;      // Forward NTT twiddles (bit-reversed storage)
    std::vector<uint32_t> inverse;      // Inverse NTT twiddles

    uint32_t N;
    uint32_t q;

    static GPU32TwiddleTable create(const GPU32Prime& prime);
};

inline GPU32TwiddleTable GPU32TwiddleTable::create(const GPU32Prime& prime) {
    GPU32TwiddleTable table;
    table.N = prime.N;
    table.q = prime.q;
    table.psi_powers.resize(prime.N);
    table.psi_inv_powers.resize(prime.N);
    table.forward.resize(prime.N);
    table.inverse.resize(prime.N);

    uint32_t log_N = prime.log_N;
    uint32_t q = prime.q;

    // Bit-reverse helper
    auto bit_reverse = [](uint32_t x, uint32_t bits) -> uint32_t {
        uint32_t result = 0;
        for (uint32_t i = 0; i < bits; ++i) {
            result = (result << 1) | (x & 1);
            x >>= 1;
        }
        return result;
    };

    // Psi powers for negacyclic twist
    // psi_powers[i] = psi^i mod q
    // psi_inv_powers[i] = psi^{-i} * n_inv mod q (combined untwist + scaling)
    table.psi_powers[0] = 1;
    table.psi_inv_powers[0] = prime.n_inv;
    for (uint32_t i = 1; i < prime.N; ++i) {
        table.psi_powers[i] = static_cast<uint32_t>(
            (static_cast<uint64_t>(table.psi_powers[i-1]) * prime.psi) % q);
        table.psi_inv_powers[i] = static_cast<uint32_t>(
            (static_cast<uint64_t>(table.psi_inv_powers[i-1]) * prime.psi_inv) % q);
    }

    // Forward twiddles (Cooley-Tukey, bit-reversed storage for in-place NTT)
    // For stage s with m = 2^s groups: twiddle[m + i] = omega^{(N/m) * bitrev(i)}
    for (uint32_t m = 1; m < prime.N; m <<= 1) {
        uint32_t log_m = 0;
        while ((1u << log_m) < m) ++log_m;

        for (uint32_t i = 0; i < m; ++i) {
            uint32_t exp = (prime.N / m) * bit_reverse(i, log_m);
            table.forward[m + i] = detail::powmod_32(prime.omega, exp, q);
        }
    }
    table.forward[0] = 1;

    // Inverse twiddles (Gentleman-Sande)
    for (uint32_t m = 1; m < prime.N; m <<= 1) {
        uint32_t log_m = 0;
        while ((1u << log_m) < m) ++log_m;

        for (uint32_t i = 0; i < m; ++i) {
            uint32_t exp = (prime.N / m) * bit_reverse(i, log_m);
            table.inverse[m + i] = detail::powmod_32(prime.omega_inv, exp, q);
        }
    }
    table.inverse[0] = 1;

    return table;
}

// =============================================================================
// GPUPrimeSet - Collection of primes for RNS-based FHE
// =============================================================================
//
// NTT-friendly 32-bit primes selected for GPU efficiency:
// - All primes Q satisfy Q = 1 (mod 16384) to support N up to 8192
// - Primes are near 2^30-2^32 for maximum precision
// - Barrett reduction fits in 64-bit arithmetic
//
// Selection process:
// 1. Start from 2^32 - 1 and search downward
// 2. Check Q = 1 (mod 16384)
// 3. Verify primality via Miller-Rabin
// 4. Verify primitive 2N-th root exists

class GPUPrimeSet {
public:
    // Maximum supported ring dimension
    static constexpr uint32_t MAX_N = 8192;
    static constexpr uint32_t MIN_N = 1024;

    // Get primes suitable for ring dimension N
    // Returns up to 'count' primes, throws if insufficient primes available
    static std::vector<GPU32Prime> get_primes_for_n(uint32_t N, size_t count);

    // Get a single prime (convenience wrapper)
    static GPU32Prime get_prime_for_n(uint32_t N, size_t index = 0);

    // Get precomputed twiddle tables for a prime
    static GPU32TwiddleTable get_twiddles(const GPU32Prime& prime);

    // Number of available primes for a given N
    static size_t num_primes_for_n(uint32_t N);

    // Verify all primes in the set
    static bool verify_all();

private:
    // Static prime tables (lazily initialized)
    static const std::vector<uint32_t>& get_prime_table();

    // Cache for twiddle tables (keyed by q|N)
    static std::unordered_map<uint64_t, GPU32TwiddleTable>& twiddle_cache();
};

// =============================================================================
// Curated Prime List
// =============================================================================
//
// These primes were selected to satisfy:
// - Q = 1 (mod 16384) for N up to 8192
// - Q in [2^30, 2^32) for 32-bit GPU efficiency
// - Verified prime via Miller-Rabin
// - Diverse bit patterns for RNS stability

inline const std::vector<uint32_t>& GPUPrimeSet::get_prime_table() {
    // Static storage for prime list
    // All primes Q satisfy Q = 1 (mod 16384)
    // Verified via Miller-Rabin with witnesses {2, 7, 61}
    // Sorted in descending order for maximum precision first
    static const std::vector<uint32_t> primes = {
        // Near 2^32 (maximum 32-bit precision)
        // These primes are Q = 1 + k * 16384 for various k
        4294475777u,   // 1 + 262114 * 16384 (verified prime)
        4293918721u,   // 1 + 262080 * 16384 (verified prime)
        4293836801u,   // 1 + 262075 * 16384 (verified prime)
        4293230593u,   // 1 + 262038 * 16384 (verified prime)
        4293181441u,   // 1 + 262035 * 16384 (verified prime)
        4292984833u,   // 1 + 262023 * 16384 (verified prime)
        4292804609u,   // 1 + 262012 * 16384 (verified prime)
        4292755457u,   // 1 + 262009 * 16384 (verified prime)

        // Near 2^31 (good balance)
        2147352577u,   // 1 + 131064 * 16384 (verified prime)
        2147205121u,   // 1 + 131055 * 16384 (verified prime)
        2147074049u,   // 1 + 131047 * 16384 (verified prime)
        2146959361u,   // 1 + 131040 * 16384 (verified prime)
        2146713601u,   // 1 + 131025 * 16384 (verified prime)
        2146418689u,   // 1 + 131007 * 16384 (verified prime)
        2146336769u,   // 1 + 131002 * 16384 (verified prime)
        2146091009u,   // 1 + 130987 * 16384 (verified prime)

        // Near 2^30 (smaller primes for variety)
        1073692673u,   // 1 + 65533 * 16384 (verified prime)
        1073643521u,   // 1 + 65530 * 16384 (verified prime)
        1073479681u,   // 1 + 65520 * 16384 (verified prime)
        1073430529u,   // 1 + 65517 * 16384 (verified prime)
        1073299457u,   // 1 + 65509 * 16384 (verified prime)
        1073233921u,   // 1 + 65505 * 16384 (verified prime)
        1073184769u,   // 1 + 65502 * 16384 (verified prime)
        1073135617u,   // 1 + 65499 * 16384 (verified prime)
    };

    return primes;
}

inline std::unordered_map<uint64_t, GPU32TwiddleTable>& GPUPrimeSet::twiddle_cache() {
    static std::unordered_map<uint64_t, GPU32TwiddleTable> cache;
    return cache;
}

inline std::vector<GPU32Prime> GPUPrimeSet::get_primes_for_n(uint32_t N, size_t count) {
    if (N < MIN_N || N > MAX_N) {
        throw std::invalid_argument("N must be in [1024, 8192]");
    }
    if ((N & (N - 1)) != 0) {
        throw std::invalid_argument("N must be power of 2");
    }

    const auto& prime_table = get_prime_table();
    std::vector<GPU32Prime> result;
    result.reserve(count);

    for (uint32_t q : prime_table) {
        if (result.size() >= count) break;

        // Check Q = 1 (mod 2N)
        if ((q - 1) % (2 * N) != 0) continue;

        // Verify primality (should always pass for our curated list)
        if (!detail::is_prime_32(q)) continue;

        try {
            GPU32Prime prime = GPU32Prime::create(q, N);
            if (prime.verify()) {
                result.push_back(prime);
            }
        } catch (...) {
            // Skip primes that fail construction
            continue;
        }
    }

    if (result.size() < count) {
        throw std::runtime_error("Insufficient primes available for N=" +
                                  std::to_string(N) + ", requested=" +
                                  std::to_string(count) + ", found=" +
                                  std::to_string(result.size()));
    }

    return result;
}

inline GPU32Prime GPUPrimeSet::get_prime_for_n(uint32_t N, size_t index) {
    auto primes = get_primes_for_n(N, index + 1);
    return primes[index];
}

inline GPU32TwiddleTable GPUPrimeSet::get_twiddles(const GPU32Prime& prime) {
    uint64_t key = (static_cast<uint64_t>(prime.q) << 32) | prime.N;

    auto& cache = twiddle_cache();
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    GPU32TwiddleTable table = GPU32TwiddleTable::create(prime);
    cache[key] = table;
    return table;
}

inline size_t GPUPrimeSet::num_primes_for_n(uint32_t N) {
    if (N < MIN_N || N > MAX_N) return 0;
    if ((N & (N - 1)) != 0) return 0;

    const auto& prime_table = get_prime_table();
    size_t count = 0;

    for (uint32_t q : prime_table) {
        if ((q - 1) % (2 * N) == 0 && detail::is_prime_32(q)) {
            ++count;
        }
    }

    return count;
}

inline bool GPUPrimeSet::verify_all() {
    const auto& prime_table = get_prime_table();

    for (uint32_t q : prime_table) {
        if (!detail::is_prime_32(q)) return false;
        if ((q - 1) % 16384 != 0) return false;  // Must support N=8192
    }

    // Verify we have enough primes for each N
    for (uint32_t N = MIN_N; N <= MAX_N; N <<= 1) {
        if (num_primes_for_n(N) < 4) return false;
    }

    return true;
}

// =============================================================================
// Convenience: Static Prime Sets per N
// =============================================================================
//
// Pre-validated prime configurations for common ring dimensions.

namespace primes {

// N=1024: 10 stages, supports up to 2^10 = 1024 coefficients
// Q = 1 (mod 2048)
struct N1024 {
    static constexpr uint32_t N = 1024;
    static constexpr uint32_t LOG_N = 10;

    // Top 8 primes for N=1024
    static std::array<GPU32Prime, 8> get() {
        static bool initialized = false;
        static std::array<GPU32Prime, 8> primes;

        if (!initialized) {
            auto vec = GPUPrimeSet::get_primes_for_n(N, 8);
            std::copy_n(vec.begin(), 8, primes.begin());
            initialized = true;
        }

        return primes;
    }
};

// N=2048: 11 stages
// Q = 1 (mod 4096)
struct N2048 {
    static constexpr uint32_t N = 2048;
    static constexpr uint32_t LOG_N = 11;

    static std::array<GPU32Prime, 8> get() {
        static bool initialized = false;
        static std::array<GPU32Prime, 8> primes;

        if (!initialized) {
            auto vec = GPUPrimeSet::get_primes_for_n(N, 8);
            std::copy_n(vec.begin(), 8, primes.begin());
            initialized = true;
        }

        return primes;
    }
};

// N=4096: 12 stages
// Q = 1 (mod 8192)
struct N4096 {
    static constexpr uint32_t N = 4096;
    static constexpr uint32_t LOG_N = 12;

    static std::array<GPU32Prime, 8> get() {
        static bool initialized = false;
        static std::array<GPU32Prime, 8> primes;

        if (!initialized) {
            auto vec = GPUPrimeSet::get_primes_for_n(N, 8);
            std::copy_n(vec.begin(), 8, primes.begin());
            initialized = true;
        }

        return primes;
    }
};

// N=8192: 13 stages
// Q = 1 (mod 16384)
struct N8192 {
    static constexpr uint32_t N = 8192;
    static constexpr uint32_t LOG_N = 13;

    static std::array<GPU32Prime, 8> get() {
        static bool initialized = false;
        static std::array<GPU32Prime, 8> primes;

        if (!initialized) {
            auto vec = GPUPrimeSet::get_primes_for_n(N, 8);
            std::copy_n(vec.begin(), 8, primes.begin());
            initialized = true;
        }

        return primes;
    }
};

}  // namespace primes

// =============================================================================
// GPU32NTT - Lightweight NTT using 32-bit primes
// =============================================================================
//
// Reference CPU implementation for correctness verification.
// GPU kernels should implement the same algorithm.
//
// Supports two modes:
// 1. Negacyclic NTT: For polynomial multiplication in Z[x]/(x^N + 1)
//    - Used in FHE (RLWE/RGSW encryption)
//    - forward_negacyclic / inverse_negacyclic
//
// 2. Standard NTT: For polynomial multiplication in Z[x]/(x^N - 1)
//    - forward / inverse
//
// The negacyclic NTT is implemented as:
//    Forward: twist by psi, then standard NTT
//    Inverse: standard INTT, then untwist by psi^{-1}

class GPU32NTT {
public:
    explicit GPU32NTT(const GPU32Prime& prime)
        : prime_(prime), twiddles_(GPUPrimeSet::get_twiddles(prime)) {}

    // =========================================================================
    // Standard NTT (for x^N - 1 ring)
    // =========================================================================

    // Forward NTT (Cooley-Tukey, in-place, with bit-reversal)
    void forward(std::vector<uint32_t>& data) const {
        if (data.size() != prime_.N) {
            throw std::invalid_argument("Data size must match N");
        }

        uint32_t N = prime_.N;

        // Bit-reverse permutation
        for (uint32_t i = 0; i < N; ++i) {
            uint32_t j = bit_reverse(i, prime_.log_N);
            if (i < j) std::swap(data[i], data[j]);
        }

        // Cooley-Tukey butterflies
        for (uint32_t len = 2; len <= N; len <<= 1) {
            uint32_t half = len >> 1;
            uint32_t omega_step = N / len;

            for (uint32_t i = 0; i < N; i += len) {
                uint32_t w_idx = 0;
                for (uint32_t j = 0; j < half; ++j) {
                    uint32_t w = detail::powmod_32(prime_.omega, w_idx, prime_.q);
                    uint32_t u = data[i + j];
                    uint32_t v = prime_.mulmod(data[i + j + half], w);
                    data[i + j] = prime_.addmod(u, v);
                    data[i + j + half] = prime_.submod(u, v);
                    w_idx += omega_step;
                }
            }
        }
    }

    // Inverse NTT (Gentleman-Sande, in-place)
    void inverse(std::vector<uint32_t>& data) const {
        if (data.size() != prime_.N) {
            throw std::invalid_argument("Data size must match N");
        }

        uint32_t N = prime_.N;

        // Gentleman-Sande butterflies
        for (uint32_t len = N; len >= 2; len >>= 1) {
            uint32_t half = len >> 1;
            uint32_t omega_step = N / len;

            for (uint32_t i = 0; i < N; i += len) {
                uint32_t w_idx = 0;
                for (uint32_t j = 0; j < half; ++j) {
                    uint32_t w = detail::powmod_32(prime_.omega_inv, w_idx, prime_.q);
                    uint32_t u = data[i + j];
                    uint32_t v = data[i + j + half];
                    data[i + j] = prime_.addmod(u, v);
                    data[i + j + half] = prime_.mulmod(prime_.submod(u, v), w);
                    w_idx += omega_step;
                }
            }
        }

        // Bit-reverse permutation
        for (uint32_t i = 0; i < N; ++i) {
            uint32_t j = bit_reverse(i, prime_.log_N);
            if (i < j) std::swap(data[i], data[j]);
        }

        // Scale by N^{-1}
        for (uint32_t i = 0; i < N; ++i) {
            data[i] = prime_.mulmod(data[i], prime_.n_inv);
        }
    }

    // =========================================================================
    // Negacyclic NTT (for x^N + 1 ring, used in FHE)
    // =========================================================================

    // Forward negacyclic NTT
    // Transforms polynomial a(x) for multiplication in Z_q[x]/(x^N + 1)
    void forward_negacyclic(std::vector<uint32_t>& data) const {
        if (data.size() != prime_.N) {
            throw std::invalid_argument("Data size must match N");
        }

        uint32_t N = prime_.N;

        // Step 1: Twist by psi^i (converts to standard NTT domain)
        for (uint32_t i = 0; i < N; ++i) {
            data[i] = prime_.mulmod(data[i], twiddles_.psi_powers[i]);
        }

        // Step 2: Standard forward NTT
        forward(data);
    }

    // Inverse negacyclic NTT
    void inverse_negacyclic(std::vector<uint32_t>& data) const {
        if (data.size() != prime_.N) {
            throw std::invalid_argument("Data size must match N");
        }

        uint32_t N = prime_.N;

        // Step 1: Standard inverse NTT (without final N^{-1} scaling)
        // We'll combine the scaling with untwist for efficiency

        // GS butterflies
        for (uint32_t len = N; len >= 2; len >>= 1) {
            uint32_t half = len >> 1;
            uint32_t omega_step = N / len;

            for (uint32_t i = 0; i < N; i += len) {
                uint32_t w_idx = 0;
                for (uint32_t j = 0; j < half; ++j) {
                    uint32_t w = detail::powmod_32(prime_.omega_inv, w_idx, prime_.q);
                    uint32_t u = data[i + j];
                    uint32_t v = data[i + j + half];
                    data[i + j] = prime_.addmod(u, v);
                    data[i + j + half] = prime_.mulmod(prime_.submod(u, v), w);
                    w_idx += omega_step;
                }
            }
        }

        // Bit-reverse permutation
        for (uint32_t i = 0; i < N; ++i) {
            uint32_t j = bit_reverse(i, prime_.log_N);
            if (i < j) std::swap(data[i], data[j]);
        }

        // Step 2: Untwist by psi^{-i} and scale by N^{-1} (combined in psi_inv_powers)
        for (uint32_t i = 0; i < N; ++i) {
            data[i] = prime_.mulmod(data[i], twiddles_.psi_inv_powers[i]);
        }
    }

    // =========================================================================
    // Polynomial operations
    // =========================================================================

    // Pointwise multiplication mod q (for NTT-domain polynomials)
    void pointwise_mul(const std::vector<uint32_t>& a,
                       const std::vector<uint32_t>& b,
                       std::vector<uint32_t>& result) const {
        if (a.size() != prime_.N || b.size() != prime_.N) {
            throw std::invalid_argument("Input sizes must match N");
        }
        result.resize(prime_.N);
        for (uint32_t i = 0; i < prime_.N; ++i) {
            result[i] = prime_.mulmod(a[i], b[i]);
        }
    }

    // Polynomial multiplication in Z_q[x]/(x^N + 1) using negacyclic NTT
    void poly_mul_negacyclic(std::vector<uint32_t>& a,
                              std::vector<uint32_t>& b,
                              std::vector<uint32_t>& result) const {
        forward_negacyclic(a);
        forward_negacyclic(b);
        pointwise_mul(a, b, result);
        inverse_negacyclic(result);
    }

    const GPU32Prime& prime() const { return prime_; }
    const GPU32TwiddleTable& twiddles() const { return twiddles_; }

private:
    GPU32Prime prime_;
    GPU32TwiddleTable twiddles_;

    // Bit-reverse helper
    static uint32_t bit_reverse(uint32_t x, uint32_t bits) {
        uint32_t result = 0;
        for (uint32_t i = 0; i < bits; ++i) {
            result = (result << 1) | (x & 1);
            x >>= 1;
        }
        return result;
    }
};

}  // namespace gpu
}  // namespace lux::fhe

#endif  // LUX_FHE_MATH_HAL_MLX_GPU_PRIMES_H
