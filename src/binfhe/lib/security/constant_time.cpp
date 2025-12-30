// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// Constant-time operations implementation
// Contains non-inline and verification functions

#include "security/constant_time.h"
#include <array>
#include <chrono>
#include <stdexcept>
#include <vector>

namespace lbcrypto {
namespace security {

// ============================================================================
// Verification and Testing Utilities
// ============================================================================

/**
 * @brief Verify that a constant-time lookup implementation is correct
 *
 * This is a runtime self-test to ensure the ct_lookup function
 * works correctly. Should be called during initialization.
 *
 * @return true if verification passes
 */
bool VerifyConstantTimeLookup() {
    constexpr size_t TABLE_SIZE = 16;
    std::array<uint64_t, TABLE_SIZE> table;

    // Fill with distinct values
    for (size_t i = 0; i < TABLE_SIZE; ++i) {
        table[i] = 0x100 + i * 0x11;
    }

    // Verify each lookup returns the correct value
    for (size_t i = 0; i < TABLE_SIZE; ++i) {
        uint64_t result = ct_lookup(table.data(), TABLE_SIZE, i);
        if (result != table[i]) {
            return false;
        }
    }

    return true;
}

/**
 * @brief Verify constant-time comparison operations
 *
 * @return true if all comparisons are correct
 */
bool VerifyConstantTimeCompare() {
    // Test ct_eq
    if (ct_eq<uint64_t>(5, 5) != ~static_cast<uint64_t>(0)) return false;
    if (ct_eq<uint64_t>(5, 6) != 0) return false;
    if (ct_eq<uint64_t>(0, 0) != ~static_cast<uint64_t>(0)) return false;

    // Test ct_lt
    if (ct_lt<uint64_t>(3, 5) != ~static_cast<uint64_t>(0)) return false;
    if (ct_lt<uint64_t>(5, 3) != 0) return false;
    if (ct_lt<uint64_t>(5, 5) != 0) return false;
    if (ct_lt<uint64_t>(0, 1) != ~static_cast<uint64_t>(0)) return false;

    // Edge cases near overflow
    constexpr uint64_t MAX = ~static_cast<uint64_t>(0);
    if (ct_lt<uint64_t>(MAX - 1, MAX) != ~static_cast<uint64_t>(0)) return false;
    if (ct_lt<uint64_t>(MAX, MAX - 1) != 0) return false;
    if (ct_lt<uint64_t>(0, MAX) != ~static_cast<uint64_t>(0)) return false;

    // Test ct_select
    if (ct_select<uint64_t>(~static_cast<uint64_t>(0), 10, 20) != 10) return false;
    if (ct_select<uint64_t>(0, 10, 20) != 20) return false;

    return true;
}

/**
 * @brief Verify constant-time memory operations
 *
 * @return true if all memory operations are correct
 */
bool VerifyConstantTimeMemory() {
    std::array<uint8_t, 16> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::array<uint8_t, 16> b = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::array<uint8_t, 16> c = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 99};

    // Test ct_memcmp
    if (ct_memcmp(a.data(), b.data(), 16) != 0) return false;
    if (ct_memcmp(a.data(), c.data(), 16) == 0) return false;

    // Test ct_memcpy with mask = 0xFF (should copy)
    std::array<uint8_t, 16> dst = {0};
    ct_memcpy(dst.data(), a.data(), 16, 0xFF);
    if (ct_memcmp(dst.data(), a.data(), 16) != 0) return false;

    // Test ct_memcpy with mask = 0x00 (should not change)
    std::array<uint8_t, 16> dst2 = {0};
    ct_memcpy(dst2.data(), a.data(), 16, 0x00);
    std::array<uint8_t, 16> zeros = {0};
    if (ct_memcmp(dst2.data(), zeros.data(), 16) != 0) return false;

    // Test ct_memzero
    std::array<uint8_t, 16> to_clear = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    ct_memzero(to_clear.data(), 16);
    if (ct_memcmp(to_clear.data(), zeros.data(), 16) != 0) return false;

    return true;
}

/**
 * @brief Run all constant-time operation verifications
 *
 * @throws std::runtime_error if any verification fails
 */
void VerifyAllConstantTimeOps() {
    if (!VerifyConstantTimeLookup()) {
        throw std::runtime_error("Constant-time lookup verification failed");
    }
    if (!VerifyConstantTimeCompare()) {
        throw std::runtime_error("Constant-time compare verification failed");
    }
    if (!VerifyConstantTimeMemory()) {
        throw std::runtime_error("Constant-time memory verification failed");
    }
}

// ============================================================================
// Extended Lookup Table Operations
// ============================================================================

/**
 * @brief Constant-time byte array lookup
 *
 * Retrieves a fixed-size entry from a table of byte arrays.
 * All entries are accessed regardless of index.
 *
 * @param table Pointer to table of entries (flattened)
 * @param num_entries Number of entries in table
 * @param entry_size Size of each entry in bytes
 * @param secret_index Index to retrieve
 * @param output Output buffer (must be at least entry_size bytes)
 */
void ct_lookup_bytes(const uint8_t* table, size_t num_entries, size_t entry_size,
                     size_t secret_index, uint8_t* output) {
    CT_BARRIER();

    // Zero output first
    for (size_t j = 0; j < entry_size; ++j) {
        output[j] = 0;
    }

    // Access all entries, accumulating only the matching one
    for (size_t i = 0; i < num_entries; ++i) {
        uint8_t mask = static_cast<uint8_t>(ct_eq(i, secret_index));

        for (size_t j = 0; j < entry_size; ++j) {
            output[j] |= table[i * entry_size + j] & mask;
        }
    }

    CT_BARRIER();
}

/**
 * @brief Constant-time vectorized lookup for multiple indices
 *
 * Performs multiple lookups in parallel, accessing all table entries
 * once for all lookups to amortize the cost.
 *
 * @param table Lookup table
 * @param table_size Size of table
 * @param indices Array of secret indices
 * @param num_lookups Number of lookups to perform
 * @param results Output array for results
 */
void ct_lookup_batch(const uint64_t* table, size_t table_size,
                     const size_t* indices, size_t num_lookups,
                     uint64_t* results) {
    CT_BARRIER();

    // Initialize results to zero
    for (size_t k = 0; k < num_lookups; ++k) {
        results[k] = 0;
    }

    // Single pass through table, computing all lookups
    for (size_t i = 0; i < table_size; ++i) {
        uint64_t table_val = table[i];

        for (size_t k = 0; k < num_lookups; ++k) {
            auto mask = ct_eq(i, indices[k]);
            results[k] |= table_val & mask;
        }
    }

    CT_BARRIER();
}

// ============================================================================
// Constant-Time Polynomial Operations (for NTT/FFT)
// ============================================================================

/**
 * @brief Constant-time butterfly operation index selection
 *
 * In NTT/FFT butterflies, the twiddle factor index selection
 * must not leak information through cache timing.
 *
 * @param twiddles Twiddle factor table
 * @param num_twiddles Size of twiddle table
 * @param secret_idx Twiddle factor index
 * @return twiddles[secret_idx]
 */
uint64_t ct_twiddle_lookup(const uint64_t* twiddles, size_t num_twiddles,
                           size_t secret_idx) {
    return ct_lookup(twiddles, num_twiddles, secret_idx);
}

/**
 * @brief Constant-time modular reduction decision
 *
 * Performs conditional modular reduction without branching.
 * result = (val >= mod) ? val - mod : val
 *
 * @param val Value to reduce
 * @param mod Modulus
 * @return val mod mod (single reduction step)
 */
uint64_t ct_reduce_once(uint64_t val, uint64_t mod) {
    CT_BARRIER();

    // Compute mask: all 1s if val >= mod
    auto ge_mask = ct_ge(val, mod);

    // Conditionally subtract mod
    uint64_t reduced = val - (mod & ge_mask);

    CT_BARRIER();
    return reduced;
}

/**
 * @brief Constant-time centered reduction
 *
 * Maps val in [0, mod) to [-mod/2, mod/2)
 * Used in decryption for proper rounding.
 *
 * @param val Value in [0, mod)
 * @param mod Modulus
 * @param half mod/2 (precomputed for efficiency)
 * @return Centered value
 */
int64_t ct_center(uint64_t val, uint64_t mod, uint64_t half) {
    CT_BARRIER();

    // If val > half, return val - mod (as signed)
    // Else return val
    auto gt_half = ct_gt(val, half);

    // Use signed arithmetic carefully
    int64_t centered = static_cast<int64_t>(val);
    int64_t mod_signed = static_cast<int64_t>(mod);

    // Conditionally subtract mod
    int64_t adjustment = mod_signed & static_cast<int64_t>(gt_half);
    centered -= adjustment;

    CT_BARRIER();
    return centered;
}

// ============================================================================
// Constant-Time Comparison Prefix Scan (Kogge-Stone)
// ============================================================================

/**
 * @brief Parallel prefix scan for radix integer comparison
 *
 * Performs Kogge-Stone parallel prefix computation on comparison flags.
 * All operations are constant-time with uniform memory access.
 *
 * @param flags Input comparison flags (modified in place)
 * @param num_limbs Number of limbs
 */
void ct_prefix_compare(uint8_t* flags, size_t num_limbs) {
    if (num_limbs <= 1) return;

    CT_BARRIER();

    // Kogge-Stone prefix scan
    // At each level, combine pairs with increasing stride
    for (size_t stride = 1; stride < num_limbs; stride *= 2) {
        // Process all pairs at this level
        // Use temp array to avoid race conditions
        std::vector<uint8_t> temp(num_limbs);

        for (size_t i = 0; i < num_limbs; ++i) {
            if (i >= stride) {
                temp[i] = ct_combine_flags(flags[i], flags[i - stride]);
            } else {
                temp[i] = flags[i];
            }
        }

        // Copy back
        for (size_t i = 0; i < num_limbs; ++i) {
            flags[i] = temp[i];
        }
    }

    CT_BARRIER();
}

/**
 * @brief Constant-time radix integer comparison
 *
 * Compares two radix integers in constant time.
 * All limbs are accessed regardless of comparison result.
 *
 * @param a First integer (array of limbs, little-endian)
 * @param b Second integer
 * @param num_limbs Number of limbs
 * @return Comparison flag: 0=LT, 1=EQ, 2=GT
 */
uint8_t ct_radix_compare(const uint64_t* a, const uint64_t* b, size_t num_limbs) {
    CT_BARRIER();

    // Compute per-limb comparison flags
    std::vector<uint8_t> flags(num_limbs);
    for (size_t i = 0; i < num_limbs; ++i) {
        flags[i] = ct_compare_flag(a[i], b[i]);
    }

    // Prefix scan from high limb to low
    // flags[i] represents comparison result considering limbs i..num_limbs-1

    // Reverse for high-to-low processing
    std::vector<uint8_t> rev_flags(num_limbs);
    for (size_t i = 0; i < num_limbs; ++i) {
        rev_flags[i] = flags[num_limbs - 1 - i];
    }

    ct_prefix_compare(rev_flags.data(), num_limbs);

    // Final result is at index num_limbs - 1 (originally limb 0)
    uint8_t result = rev_flags[num_limbs - 1];

    CT_BARRIER();
    return result;
}

// ============================================================================
// Constant-Time LWE Operations
// ============================================================================

/**
 * @brief Constant-time LWE sample addition with bounds check
 *
 * Adds two LWE ciphertext vectors with constant-time modular reduction.
 * No early exit on overflow.
 *
 * @param result Output vector
 * @param a First ciphertext
 * @param b Second ciphertext
 * @param n Vector dimension
 * @param mod Modulus
 */
void ct_lwe_add(uint64_t* result, const uint64_t* a, const uint64_t* b,
                size_t n, uint64_t mod) {
    CT_BARRIER();

    for (size_t i = 0; i <= n; ++i) {  // n+1 elements (n vector + 1 constant term)
        uint64_t sum = a[i] + b[i];
        result[i] = ct_reduce_once(sum, mod);
    }

    CT_BARRIER();
}

/**
 * @brief Constant-time LWE sample subtraction
 */
void ct_lwe_sub(uint64_t* result, const uint64_t* a, const uint64_t* b,
                size_t n, uint64_t mod) {
    CT_BARRIER();

    for (size_t i = 0; i <= n; ++i) {
        // (a - b) mod q = (a + (q - b)) mod q
        uint64_t neg_b = mod - b[i];
        neg_b = ct_reduce_once(neg_b, mod);  // Handle b[i] == 0

        uint64_t sum = a[i] + neg_b;
        result[i] = ct_reduce_once(sum, mod);
    }

    CT_BARRIER();
}

/**
 * @brief Constant-time scalar multiplication
 */
void ct_lwe_scalar_mul(uint64_t* result, const uint64_t* a, uint64_t scalar,
                       size_t n, uint64_t mod) {
    CT_BARRIER();

    for (size_t i = 0; i <= n; ++i) {
        // Use 128-bit intermediate for full precision
        __uint128_t prod = static_cast<__uint128_t>(a[i]) * scalar;
        result[i] = static_cast<uint64_t>(prod % mod);
    }

    CT_BARRIER();
}

// ============================================================================
// Constant-Time Random Sampling Helpers
// ============================================================================

/**
 * @brief Constant-time rejection sampling decision
 *
 * Returns 1 if sample should be accepted, 0 if rejected.
 * The decision is made in constant time.
 *
 * @param sample Random sample
 * @param bound Rejection bound
 * @return Acceptance mask (~0 if accept, 0 if reject)
 */
uint64_t ct_rejection_decision(uint64_t sample, uint64_t bound) {
    // Accept if sample < bound
    return ct_lt(sample, bound);
}

/**
 * @brief Constant-time conditional sample replacement
 *
 * Replaces current sample with new sample if current was rejected.
 * Used in rejection sampling to avoid timing leaks.
 *
 * @param current Current sample
 * @param current_valid Mask indicating if current is valid
 * @param new_sample New candidate sample
 * @param new_valid Mask indicating if new is valid
 * @return Updated sample
 */
uint64_t ct_sample_update(uint64_t current, uint64_t current_valid,
                          uint64_t new_sample, uint64_t new_valid) {
    CT_BARRIER();

    // If current is invalid but new is valid, use new
    auto should_update = (~current_valid) & new_valid;
    uint64_t result = ct_select(should_update, new_sample, current);

    CT_BARRIER();
    return result;
}

} // namespace security
} // namespace lbcrypto
