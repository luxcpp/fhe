// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// Constant-time operations for side-channel resistance
// All operations have data-independent timing and memory access patterns
//
// Design Principles:
// 1. No conditional branches on secret data
// 2. No early-exit comparisons
// 3. Uniform memory access patterns (touch all elements)
// 4. Compiler barrier to prevent optimization of security-critical code
// 5. All operations verified with TIMECOP or similar analysis tools
//
// References:
// - "Timing Attacks on Implementations of Diffie-Hellman, RSA, DSS, and Other Systems" (Kocher 1996)
// - "Cache-timing attacks on AES" (Bernstein 2005)
// - "ctgrind" (Langley, https://github.com/agl/ctgrind)

#ifndef LUX_FHE_SECURITY_CONSTANT_TIME_H
#define LUX_FHE_SECURITY_CONSTANT_TIME_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

// Compiler barrier to prevent optimization of constant-time operations
// This prevents the compiler from "helpfully" optimizing our security measures
#if defined(__GNUC__) || defined(__clang__)
    #define CT_BARRIER() __asm__ __volatile__("" ::: "memory")
    #define CT_LIKELY(x) __builtin_expect(!!(x), 1)
    #define CT_UNLIKELY(x) __builtin_expect(!!(x), 0)
#elif defined(_MSC_VER)
    #include <intrin.h>
    #define CT_BARRIER() _ReadWriteBarrier()
    #define CT_LIKELY(x) (x)
    #define CT_UNLIKELY(x) (x)
#else
    #define CT_BARRIER() do {} while(0)
    #define CT_LIKELY(x) (x)
    #define CT_UNLIKELY(x) (x)
#endif

// Mark a value as "secret" to prevent compiler optimizations
// Use volatile_load/store to ensure the compiler doesn't optimize away accesses
#if defined(__GNUC__) || defined(__clang__)
    #define CT_VOLATILE_LOAD(x) (*((volatile typeof(x)*)&(x)))
    #define CT_VOLATILE_STORE(ptr, val) (*((volatile typeof(val)*)(ptr)) = (val))
#else
    #define CT_VOLATILE_LOAD(x) (x)
    #define CT_VOLATILE_STORE(ptr, val) (*(ptr) = (val))
#endif

namespace lbcrypto {
namespace security {

// ============================================================================
// Core Constant-Time Primitives
// ============================================================================

/**
 * @brief Constant-time conditional select
 *
 * Returns a if mask is all 1s, b if mask is all 0s.
 * The mask MUST be either 0x00...00 or 0xFF...FF.
 *
 * @param mask Selection mask (0 = select b, ~0 = select a)
 * @param a Value to return if mask is all 1s
 * @param b Value to return if mask is all 0s
 * @return a if mask is ~0, b if mask is 0
 *
 * Implementation: result = (a & mask) | (b & ~mask)
 * This executes in constant time with no branches.
 */
template<typename T>
inline T ct_select(T mask, T a, T b) noexcept {
    static_assert(std::is_integral<T>::value, "ct_select requires integral type");
    CT_BARRIER();
    T result = (a & mask) | (b & ~mask);
    CT_BARRIER();
    return result;
}

/**
 * @brief Constant-time boolean select (convenience wrapper)
 *
 * @param condition Boolean condition (true = select a, false = select b)
 * @param a Value to return if condition is true
 * @param b Value to return if condition is false
 * @return a if condition is true, b otherwise
 */
template<typename T>
inline T ct_select_bool(bool condition, T a, T b) noexcept {
    static_assert(std::is_integral<T>::value, "ct_select_bool requires integral type");
    // Convert bool to mask: true -> ~0, false -> 0
    // Important: use arithmetic, not branches
    T mask = static_cast<T>(-(static_cast<T>(condition)));
    return ct_select(mask, a, b);
}

/**
 * @brief Constant-time equality check
 *
 * Returns a mask of all 1s if a == b, all 0s otherwise.
 * Uses XOR and arithmetic to avoid branches.
 *
 * @param a First value
 * @param b Second value
 * @return ~0 if a == b, 0 otherwise
 */
template<typename T>
inline T ct_eq(T a, T b) noexcept {
    static_assert(std::is_unsigned<T>::value, "ct_eq requires unsigned type");
    CT_BARRIER();

    // XOR gives 0 iff equal
    T diff = a ^ b;

    // Collapse diff to single bit: if any bit is set, result is non-zero
    // Use: (diff - 1) >> (bits - 1) flips high bit iff diff was 0
    // But simpler: ((diff | -diff) >> (bits - 1)) gives 1 if diff != 0
    // So we want the inverse

    // This technique: if diff == 0, (diff | -diff) == 0
    // Otherwise, high bit of (diff | -diff) is 1
    // Right shift gives us 0 or 1
    constexpr size_t bits = sizeof(T) * 8;
    // Note: Cast to T before shift to avoid integer promotion issues
    // (e.g., uint8_t gets promoted to int, causing wrong shift results)
    T is_nonzero = static_cast<T>(diff | (~diff + 1)) >> (bits - 1);

    // is_nonzero is 0 if equal, 1 if not equal
    // We want mask of all 1s if equal
    T result = ~(static_cast<T>(0) - is_nonzero);

    CT_BARRIER();
    return result;
}

/**
 * @brief Constant-time "is zero" check
 *
 * @param a Value to check
 * @return ~0 if a == 0, 0 otherwise
 */
template<typename T>
inline T ct_is_zero(T a) noexcept {
    static_assert(std::is_unsigned<T>::value, "ct_is_zero requires unsigned type");
    return ct_eq<T>(a, static_cast<T>(0));
}

/**
 * @brief Constant-time "is non-zero" check
 *
 * @param a Value to check
 * @return ~0 if a != 0, 0 otherwise
 */
template<typename T>
inline T ct_is_nonzero(T a) noexcept {
    static_assert(std::is_unsigned<T>::value, "ct_is_nonzero requires unsigned type");
    return ~ct_is_zero(a);
}

/**
 * @brief Constant-time less-than comparison
 *
 * Returns mask of all 1s if a < b, all 0s otherwise.
 * Uses the "borrow" from subtraction to determine result.
 *
 * @param a First value
 * @param b Second value
 * @return ~0 if a < b, 0 otherwise
 */
template<typename T>
inline T ct_lt(T a, T b) noexcept {
    static_assert(std::is_unsigned<T>::value, "ct_lt requires unsigned type");
    CT_BARRIER();

    constexpr size_t bits = sizeof(T) * 8;

    // For unsigned comparison: a < b iff high bit of (a - b) is set when no wrap
    // But we need to handle the wrap case. Use:
    // (a ^ ((a ^ b) | ((a - b) ^ b))) >> (bits - 1)
    // This correctly handles all cases
    T diff = a - b;
    T lt_bit = (a ^ ((a ^ b) | (diff ^ b))) >> (bits - 1);

    // Extend single bit to full mask
    T result = static_cast<T>(0) - lt_bit;

    CT_BARRIER();
    return result;
}

/**
 * @brief Constant-time less-than-or-equal comparison
 */
template<typename T>
inline T ct_le(T a, T b) noexcept {
    return ct_lt(a, b) | ct_eq(a, b);
}

/**
 * @brief Constant-time greater-than comparison
 */
template<typename T>
inline T ct_gt(T a, T b) noexcept {
    return ct_lt(b, a);
}

/**
 * @brief Constant-time greater-than-or-equal comparison
 */
template<typename T>
inline T ct_ge(T a, T b) noexcept {
    return ct_le(b, a);
}

// ============================================================================
// Constant-Time Memory Operations
// ============================================================================

/**
 * @brief Constant-time memory comparison
 *
 * Compares n bytes of memory, returning 0 if equal, non-zero otherwise.
 * ALWAYS reads all n bytes regardless of differences found.
 * No early exit.
 *
 * @param a First memory region
 * @param b Second memory region
 * @param n Number of bytes to compare
 * @return 0 if equal, non-zero otherwise
 */
inline int ct_memcmp(const void* a, const void* b, size_t n) noexcept {
    const volatile uint8_t* pa = static_cast<const volatile uint8_t*>(a);
    const volatile uint8_t* pb = static_cast<const volatile uint8_t*>(b);

    CT_BARRIER();

    uint8_t diff = 0;
    for (size_t i = 0; i < n; ++i) {
        diff |= pa[i] ^ pb[i];
    }

    CT_BARRIER();
    return static_cast<int>(diff);
}

/**
 * @brief Constant-time conditional memory copy
 *
 * Copies src to dst if mask is all 1s, otherwise dst unchanged.
 * ALWAYS reads all len bytes from both src and dst.
 * ALWAYS writes len bytes to dst (either new or same value).
 *
 * @param dst Destination buffer
 * @param src Source buffer
 * @param len Number of bytes
 * @param mask Selection mask (~0 = copy, 0 = no-op, but still touch memory)
 */
inline void ct_memcpy(void* dst, const void* src, size_t len, uint8_t mask) noexcept {
    volatile uint8_t* pd = static_cast<volatile uint8_t*>(dst);
    const volatile uint8_t* ps = static_cast<const volatile uint8_t*>(src);

    CT_BARRIER();

    for (size_t i = 0; i < len; ++i) {
        uint8_t d = pd[i];
        uint8_t s = ps[i];
        pd[i] = ct_select(mask, s, d);
    }

    CT_BARRIER();
}

/**
 * @brief Constant-time memory clear
 *
 * Securely zeros memory. Cannot be optimized away by compiler.
 *
 * @param ptr Memory to clear
 * @param len Number of bytes
 */
inline void ct_memzero(void* ptr, size_t len) noexcept {
    volatile uint8_t* p = static_cast<volatile uint8_t*>(ptr);

    CT_BARRIER();

    for (size_t i = 0; i < len; ++i) {
        p[i] = 0;
    }

    CT_BARRIER();
}

// ============================================================================
// Constant-Time Table Lookup (Cache-Timing Resistant)
// ============================================================================

/**
 * @brief Constant-time table lookup
 *
 * Accesses ALL table entries to prevent cache-timing attacks.
 * The secret_index determines which entry's value is returned,
 * but all entries are read regardless.
 *
 * @tparam T Element type
 * @param table Pointer to lookup table
 * @param table_size Number of elements in table
 * @param secret_index Index to retrieve (must be < table_size)
 * @return table[secret_index]
 *
 * Complexity: O(table_size) - intentionally linear to prevent timing leaks
 */
template<typename T>
inline T ct_lookup(const T* table, size_t table_size, size_t secret_index) noexcept {
    static_assert(std::is_integral<T>::value, "ct_lookup requires integral type");

    CT_BARRIER();

    T result = 0;
    for (size_t i = 0; i < table_size; ++i) {
        // Create mask: all 1s if i == secret_index, else all 0s
        auto mask = ct_eq(static_cast<size_t>(i), secret_index);

        // Accumulate: only the matching entry contributes
        // result = (table[i] & mask) | (result & ~mask)
        // But since result starts at 0 and only one mask is non-zero:
        result |= table[i] & static_cast<T>(mask);
    }

    CT_BARRIER();
    return result;
}

/**
 * @brief Constant-time table lookup with explicit size type
 *
 * Variant using fixed-size index for better type safety.
 */
template<typename T, typename IndexT>
inline T ct_lookup_typed(const T* table, IndexT table_size, IndexT secret_index) noexcept {
    static_assert(std::is_integral<T>::value, "ct_lookup_typed requires integral element type");
    static_assert(std::is_unsigned<IndexT>::value, "ct_lookup_typed requires unsigned index type");

    CT_BARRIER();

    T result = 0;
    for (IndexT i = 0; i < table_size; ++i) {
        auto mask = ct_eq(i, secret_index);
        result |= table[i] & static_cast<T>(mask);
    }

    CT_BARRIER();
    return result;
}

/**
 * @brief Constant-time two-dimensional table lookup
 *
 * For LUTs indexed by two secret values (e.g., comparison tables).
 * Accesses all entries in the 2D table.
 *
 * @param table Flattened 2D table (row-major)
 * @param cols Number of columns
 * @param row Secret row index
 * @param col Secret column index
 * @return table[row * cols + col]
 */
template<typename T>
inline T ct_lookup_2d(const T* table, size_t rows, size_t cols,
                       size_t row, size_t col) noexcept {
    static_assert(std::is_integral<T>::value, "ct_lookup_2d requires integral type");

    CT_BARRIER();

    T result = 0;
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            auto row_match = ct_eq(r, row);
            auto col_match = ct_eq(c, col);
            auto mask = row_match & col_match;
            result |= table[r * cols + c] & static_cast<T>(mask);
        }
    }

    CT_BARRIER();
    return result;
}

// ============================================================================
// Constant-Time Arithmetic (Overflow-Safe)
// ============================================================================

/**
 * @brief Constant-time saturating add
 *
 * Returns a + b, or max value if overflow would occur.
 * No branches on the result.
 */
template<typename T>
inline T ct_add_sat(T a, T b) noexcept {
    static_assert(std::is_unsigned<T>::value, "ct_add_sat requires unsigned type");

    T sum = a + b;
    // Overflow occurred if sum < a (or sum < b)
    T overflow_mask = ct_lt(sum, a);

    // If overflow, return max value
    constexpr T max_val = static_cast<T>(~static_cast<T>(0));
    return ct_select(overflow_mask, max_val, sum);
}

/**
 * @brief Constant-time saturating subtract
 *
 * Returns a - b, or 0 if underflow would occur.
 */
template<typename T>
inline T ct_sub_sat(T a, T b) noexcept {
    static_assert(std::is_unsigned<T>::value, "ct_sub_sat requires unsigned type");

    T diff = a - b;
    // Underflow occurred if a < b
    T underflow_mask = ct_lt(a, b);

    return ct_select(underflow_mask, static_cast<T>(0), diff);
}

/**
 * @brief Constant-time minimum
 */
template<typename T>
inline T ct_min(T a, T b) noexcept {
    static_assert(std::is_unsigned<T>::value, "ct_min requires unsigned type");
    return ct_select(ct_lt(a, b), a, b);
}

/**
 * @brief Constant-time maximum
 */
template<typename T>
inline T ct_max(T a, T b) noexcept {
    static_assert(std::is_unsigned<T>::value, "ct_max requires unsigned type");
    return ct_select(ct_gt(a, b), a, b);
}

/**
 * @brief Constant-time clamp
 */
template<typename T>
inline T ct_clamp(T val, T min_val, T max_val) noexcept {
    return ct_min(ct_max(val, min_val), max_val);
}

// ============================================================================
// Constant-Time Bit Operations
// ============================================================================

/**
 * @brief Constant-time conditional bit set
 *
 * Sets bit at position if condition is true.
 */
template<typename T>
inline T ct_set_bit(T val, unsigned int bit_pos, bool condition) noexcept {
    static_assert(std::is_unsigned<T>::value, "ct_set_bit requires unsigned type");
    T bit_mask = static_cast<T>(1) << bit_pos;
    T cond_mask = ct_select_bool(condition, bit_mask, static_cast<T>(0));
    return val | cond_mask;
}

/**
 * @brief Constant-time conditional bit clear
 */
template<typename T>
inline T ct_clear_bit(T val, unsigned int bit_pos, bool condition) noexcept {
    static_assert(std::is_unsigned<T>::value, "ct_clear_bit requires unsigned type");
    T bit_mask = static_cast<T>(1) << bit_pos;
    T cond_mask = ct_select_bool(condition, bit_mask, static_cast<T>(0));
    return val & ~cond_mask;
}

/**
 * @brief Constant-time bit extraction
 */
template<typename T>
inline T ct_get_bit(T val, unsigned int bit_pos) noexcept {
    static_assert(std::is_unsigned<T>::value, "ct_get_bit requires unsigned type");
    return (val >> bit_pos) & static_cast<T>(1);
}

// ============================================================================
// FHE-Specific Constant-Time Operations
// ============================================================================

/**
 * @brief Constant-time comparison flag computation for radix integers
 *
 * Computes a 2-bit comparison flag (lt, eq, gt) for a pair of limbs.
 * Used in Kogge-Stone prefix scan for encrypted integer comparison.
 *
 * @param a First limb value
 * @param b Second limb value
 * @return 0 if a < b, 1 if a == b, 2 if a > b
 */
template<typename T>
inline uint8_t ct_compare_flag(T a, T b) noexcept {
    static_assert(std::is_unsigned<T>::value, "ct_compare_flag requires unsigned type");

    CT_BARRIER();

    // Compute masks for each case
    auto lt_mask = ct_lt(a, b);
    auto eq_mask = ct_eq(a, b);
    // gt is implicit: not lt and not eq

    // Build flag: 0=LT, 1=EQ, 2=GT
    // flag = (eq_mask ? 1 : 0) + (gt_mask ? 2 : 0)
    // Since lt/eq/gt are mutually exclusive:
    uint8_t flag = 0;
    flag = ct_select(static_cast<uint8_t>(eq_mask), static_cast<uint8_t>(1), flag);
    flag = ct_select(static_cast<uint8_t>(lt_mask), static_cast<uint8_t>(0), flag);

    // If neither lt nor eq, then gt -> 2
    auto gt_mask = ~(lt_mask | eq_mask);
    flag = ct_select(static_cast<uint8_t>(gt_mask), static_cast<uint8_t>(2), flag);

    CT_BARRIER();
    return flag;
}

/**
 * @brief Constant-time Kogge-Stone combine operation
 *
 * Combines two comparison flags according to Kogge-Stone semantics:
 * - If high flag indicates a definite result (LT or GT), use that
 * - If high flag is EQ, propagate the low flag
 *
 * @param flag_high Flag from higher-order limb comparison
 * @param flag_low Flag from lower-order limb comparison
 * @return Combined flag
 */
inline uint8_t ct_combine_flags(uint8_t flag_high, uint8_t flag_low) noexcept {
    CT_BARRIER();

    // If flag_high == 1 (EQ), return flag_low
    // Otherwise return flag_high
    auto high_is_eq = ct_eq(flag_high, static_cast<uint8_t>(1));
    uint8_t result = ct_select(static_cast<uint8_t>(high_is_eq), flag_low, flag_high);

    CT_BARRIER();
    return result;
}

// ============================================================================
// Verification Functions (defined in constant_time.cpp)
// ============================================================================

/**
 * @brief Verify that lookup operations are constant-time
 * @return true if constant-time, false if timing leak detected
 */
bool VerifyConstantTimeLookup();

/**
 * @brief Verify that comparison operations are constant-time
 * @return true if constant-time, false if timing leak detected
 */
bool VerifyConstantTimeCompare();

/**
 * @brief Verify that memory operations are constant-time
 * @return true if constant-time, false if timing leak detected
 */
bool VerifyConstantTimeMemory();

/**
 * @brief Run all constant-time verification tests
 * @throws std::runtime_error if any verification fails
 */
void VerifyAllConstantTimeOps();

// ============================================================================
// Radix Comparison (for multi-limb integers)
// ============================================================================

/**
 * @brief Constant-time comparison of multi-limb integers
 *
 * Compares two big integers represented as arrays of limbs.
 * Returns comparison flag using constant-time Kogge-Stone prefix scan.
 *
 * @param a First integer (array of limbs, least significant first)
 * @param b Second integer (array of limbs, least significant first)
 * @param num_limbs Number of limbs in each integer
 * @return 0 if a < b, 1 if a == b, 2 if a > b
 */
uint8_t ct_radix_compare(const uint64_t* a, const uint64_t* b, size_t num_limbs);

} // namespace security
} // namespace lbcrypto

#endif // LUX_FHE_SECURITY_CONSTANT_TIME_H
