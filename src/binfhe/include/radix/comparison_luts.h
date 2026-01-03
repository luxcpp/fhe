// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// Precomputed LUTs for efficient word-wise comparison operations
// Uses 2-bit flags (gt, eq) to enable O(log n) Kogge-Stone prefix scan

#ifndef RADIX_COMPARISON_LUTS_H
#define RADIX_COMPARISON_LUTS_H

#include "binfhecontext.h"
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace lux::fhe {
namespace radix {

/**
 * @brief Comparison flag encoding for prefix scan
 *
 * Each digit comparison produces a 2-bit flag:
 *   - FLAG_LT (0b00): a < b (gt=0, eq=0, implies lt=1)
 *   - FLAG_EQ (0b01): a == b (gt=0, eq=1)
 *   - FLAG_GT (0b10): a > b (gt=1, eq=0)
 *
 * The encoding enables efficient prefix scan: the combine operation
 * is associative and can be parallelized with Kogge-Stone.
 */
enum CompareFlag : uint8_t {
    FLAG_LT = 0b00,  // Less than (neither gt nor eq)
    FLAG_EQ = 0b01,  // Equal
    FLAG_GT = 0b10,  // Greater than
    // 0b11 is unused/invalid
};

/**
 * @brief Precomputed LUTs for comparison operations
 *
 * Strategy:
 * 1. Per-limb comparison via LUT produces 2-bit flags
 * 2. Kogge-Stone prefix scan combines flags in O(log n) depth
 * 3. Final flag extraction gives lt/eq/gt result
 *
 * This beats CRT reconstruction by avoiding expensive modular arithmetic
 * and enabling fully parallel GPU dispatch.
 */
class ComparisonLUTs {
public:
    /**
     * @brief Construct LUTs for given message bit width
     * @param message_bits Bits per limb (typically 2-4)
     */
    explicit ComparisonLUTs(uint32_t message_bits);
    ~ComparisonLUTs();

    // Non-copyable, movable
    ComparisonLUTs(const ComparisonLUTs&) = delete;
    ComparisonLUTs& operator=(const ComparisonLUTs&) = delete;
    ComparisonLUTs(ComparisonLUTs&&) noexcept;
    ComparisonLUTs& operator=(ComparisonLUTs&&) noexcept;

    // ========================================================================
    // Per-Limb Comparison LUTs
    // ========================================================================

    /**
     * @brief LUT for per-limb comparison producing 2-bit flag
     *
     * Input: packed value (a * p + b) where p = 2^message_bits
     * Output: CompareFlag (0=LT, 1=EQ, 2=GT)
     *
     * This single LUT replaces separate lt/eq/gt computations.
     */
    const std::vector<NativeInteger>& CompareFlagLUT() const;

    /**
     * @brief LUT for extracting gt bit from compare flag
     * Input: CompareFlag value (0-2)
     * Output: 1 if GT, 0 otherwise
     */
    const std::vector<NativeInteger>& ExtractGtLUT() const;

    /**
     * @brief LUT for extracting eq bit from compare flag
     * Input: CompareFlag value (0-2)
     * Output: 1 if EQ, 0 otherwise
     */
    const std::vector<NativeInteger>& ExtractEqLUT() const;

    /**
     * @brief LUT for extracting lt bit from compare flag
     * Input: CompareFlag value (0-2)
     * Output: 1 if LT, 0 otherwise
     */
    const std::vector<NativeInteger>& ExtractLtLUT() const;

    // ========================================================================
    // Kogge-Stone Prefix Combine LUTs
    // ========================================================================

    /**
     * @brief LUT for combining two comparison flags (Kogge-Stone operator)
     *
     * The combine operation for comparison flags:
     *   combine(flag_high, flag_low) = flag_high if flag_high != EQ
     *                                 = flag_low  if flag_high == EQ
     *
     * This is associative, enabling parallel prefix scan.
     *
     * Input: packed (flag_high * 4 + flag_low) using 2 bits each
     * Output: combined flag
     */
    const std::vector<NativeInteger>& CombineFlagsLUT() const;

    /**
     * @brief LUT for combining with propagate (for Kogge-Stone)
     *
     * Extended combine that also computes propagate signal:
     *   propagate = (flag_high == EQ)
     *
     * Output encodes both combined flag and propagate bit.
     */
    const std::vector<NativeInteger>& CombinePropLUT() const;

    // ========================================================================
    // Final Extraction LUTs
    // ========================================================================

    /**
     * @brief LUT to convert final flag to boolean lt result
     * Input: CompareFlag
     * Output: 1 if LT, 0 otherwise
     */
    const std::vector<NativeInteger>& FinalLtLUT() const;

    /**
     * @brief LUT to convert final flag to boolean le result
     * Input: CompareFlag
     * Output: 1 if LT or EQ, 0 otherwise
     */
    const std::vector<NativeInteger>& FinalLeLUT() const;

    /**
     * @brief LUT to convert final flag to boolean gt result
     */
    const std::vector<NativeInteger>& FinalGtLUT() const;

    /**
     * @brief LUT to convert final flag to boolean ge result
     */
    const std::vector<NativeInteger>& FinalGeLUT() const;

    // ========================================================================
    // Equality LUTs (Optimized Path)
    // ========================================================================

    /**
     * @brief LUT for per-limb equality check
     *
     * Equality has a faster path: AND all per-limb eq flags.
     * Input: packed (a * p + b)
     * Output: 1 if a == b, 0 otherwise
     */
    const std::vector<NativeInteger>& EqLimbLUT() const;

    /**
     * @brief LUT for combining equality flags (AND operation)
     * Input: packed (eq_high * 2 + eq_low)
     * Output: eq_high AND eq_low
     */
    const std::vector<NativeInteger>& CombineEqLUT() const;

    // ========================================================================
    // Accessors
    // ========================================================================

    uint32_t GetMessageBits() const { return message_bits_; }
    uint32_t GetModulus() const { return modulus_; }

private:
    uint32_t message_bits_;
    uint32_t modulus_;  // 2^message_bits

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// Static LUT Tables for Common Configurations
// ============================================================================

namespace detail {

/**
 * @brief Compile-time LUT generation for 2-bit limbs
 *
 * For 2-bit limbs (p=4), we precompute all LUTs at compile time
 * to avoid runtime initialization overhead.
 */
struct CompareLUT2Bit {
    // Per-limb comparison: input is (a*4 + b), output is flag
    // a, b in [0,3], so 16 entries
    static constexpr std::array<uint8_t, 16> COMPARE_FLAG = {{
        // b=0   b=1   b=2   b=3
        1,    2,    2,    2,    // a=0: EQ, GT, GT, GT
        0,    1,    2,    2,    // a=1: LT, EQ, GT, GT
        0,    0,    1,    2,    // a=2: LT, LT, EQ, GT
        0,    0,    0,    1,    // a=3: LT, LT, LT, EQ
    }};

    // Kogge-Stone combine: input is (high*4 + low), output is combined flag
    // high, low in [0,2] (LT, EQ, GT), so 9 entries (padded to 16)
    static constexpr std::array<uint8_t, 16> COMBINE = {{
        // low=LT  low=EQ  low=GT  (pad)
        0,       0,       0,       0,    // high=LT -> LT
        0,       1,       2,       0,    // high=EQ -> pass through low
        2,       2,       2,       0,    // high=GT -> GT
        0,       0,       0,       0,    // (padding)
    }};

    // Per-limb equality: input is (a*4 + b), output is 1 if eq
    static constexpr std::array<uint8_t, 16> EQ_LIMB = {{
        1, 0, 0, 0,  // a=0
        0, 1, 0, 0,  // a=1
        0, 0, 1, 0,  // a=2
        0, 0, 0, 1,  // a=3
    }};

    // Final extraction: input is flag, output is boolean
    static constexpr std::array<uint8_t, 4> FINAL_LT = {{1, 0, 0, 0}};  // LT=0 -> 1
    static constexpr std::array<uint8_t, 4> FINAL_LE = {{1, 1, 0, 0}};  // LT or EQ -> 1
    static constexpr std::array<uint8_t, 4> FINAL_GT = {{0, 0, 1, 0}};  // GT=2 -> 1
    static constexpr std::array<uint8_t, 4> FINAL_GE = {{0, 1, 1, 0}};  // EQ or GT -> 1
    static constexpr std::array<uint8_t, 4> FINAL_EQ = {{0, 1, 0, 0}};  // EQ=1 -> 1
    static constexpr std::array<uint8_t, 4> FINAL_NE = {{1, 0, 1, 0}};  // LT or GT -> 1
};

/**
 * @brief Compile-time LUT generation for 3-bit limbs
 */
struct CompareLUT3Bit {
    // For 3-bit limbs (p=8), we need 64 entries for two-input LUTs
    // Generated at construction time due to size
    static constexpr uint32_t MODULUS = 8;
    static constexpr uint32_t TWO_INPUT_SIZE = 64;
};

/**
 * @brief Compile-time LUT generation for 4-bit limbs
 */
struct CompareLUT4Bit {
    // For 4-bit limbs (p=16), we need 256 entries
    static constexpr uint32_t MODULUS = 16;
    static constexpr uint32_t TWO_INPUT_SIZE = 256;
};

} // namespace detail

// ============================================================================
// Kogge-Stone Depth Calculation
// ============================================================================

/**
 * @brief Calculate Kogge-Stone prefix scan depth
 * @param num_limbs Number of limbs in the radix integer
 * @return Number of parallel rounds needed
 */
constexpr uint32_t KoggeStoneLevels(uint32_t num_limbs) {
    if (num_limbs <= 1) return 0;

    uint32_t levels = 0;
    uint32_t n = num_limbs - 1;
    while (n > 0) {
        levels++;
        n >>= 1;
    }
    return levels;
}

// Compile-time verification for common sizes
static_assert(KoggeStoneLevels(4) == 2, "euint8: 4 limbs -> 2 levels");
static_assert(KoggeStoneLevels(8) == 3, "euint16: 8 limbs -> 3 levels");
static_assert(KoggeStoneLevels(16) == 4, "euint32: 16 limbs -> 4 levels");
static_assert(KoggeStoneLevels(32) == 5, "euint64: 32 limbs -> 5 levels");
static_assert(KoggeStoneLevels(64) == 6, "euint128: 64 limbs -> 6 levels");
static_assert(KoggeStoneLevels(128) == 7, "euint256: 128 limbs -> 7 levels");

} // namespace radix
} // namespace lux::fhe

#endif // RADIX_COMPARISON_LUTS_H
