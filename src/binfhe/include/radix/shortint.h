// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// ShortInt - Encrypted integers with small message space (2-16 bits)
// Foundation for radix integer arithmetic in FHE.

#ifndef RADIX_SHORTINT_H
#define RADIX_SHORTINT_H

#include "binfhecontext.h"
#include "lwe-ciphertext.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace lbcrypto {
namespace radix {

// Message space configuration
// message_bits determines the plaintext modulus: p = 2^message_bits
// carry_bits determines headroom for carries before bootstrapping
struct ShortIntParams {
    uint32_t message_bits;  // Bits per limb (typically 2-4)
    uint32_t carry_bits;    // Carry buffer bits (typically 2-3)
    
    // Total bits per limb in ciphertext = message_bits + carry_bits
    uint32_t total_bits() const { return message_bits + carry_bits; }
    
    // Maximum value before overflow
    uint64_t max_value() const { return (1ULL << message_bits) - 1; }
    
    // Carry threshold - when to propagate
    uint64_t carry_threshold() const { return 1ULL << message_bits; }
};

// Precomputed LUTs for ShortInt operations
class ShortIntLUTs {
public:
    ShortIntLUTs(const ShortIntParams& params);
    ~ShortIntLUTs();
    
    // Arithmetic LUTs
    const std::vector<NativeInteger>& AddLUT() const;
    const std::vector<NativeInteger>& AddCarryLUT() const;  // Returns carry bit
    const std::vector<NativeInteger>& SubLUT() const;
    const std::vector<NativeInteger>& SubBorrowLUT() const;
    const std::vector<NativeInteger>& MulLUT() const;       // Low bits
    const std::vector<NativeInteger>& MulHighLUT() const;   // High bits
    
    // Comparison LUTs
    const std::vector<NativeInteger>& LtLUT() const;
    const std::vector<NativeInteger>& LeLUT() const;
    const std::vector<NativeInteger>& GtLUT() const;
    const std::vector<NativeInteger>& GeLUT() const;
    const std::vector<NativeInteger>& EqLUT() const;
    const std::vector<NativeInteger>& NeLUT() const;
    
    // Bitwise LUTs
    const std::vector<NativeInteger>& AndLUT() const;
    const std::vector<NativeInteger>& OrLUT() const;
    const std::vector<NativeInteger>& XorLUT() const;
    const std::vector<NativeInteger>& NotLUT() const;
    
    // Utility LUTs
    const std::vector<NativeInteger>& IdentityLUT() const;
    const std::vector<NativeInteger>& CleanCarryLUT() const;  // Extract and clear carry
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief ShortInt - Encrypted small integer (limb)
 * 
 * This is the building block for radix integers. Each ShortInt represents
 * a single limb with message_bits of actual data and carry_bits of headroom.
 */
class ShortInt {
public:
    ShortInt() = default;
    ShortInt(BinFHEContext& cc, const ShortIntParams& params);
    ~ShortInt();
    
    // Copy/move
    ShortInt(const ShortInt& other);
    ShortInt(ShortInt&& other) noexcept;
    ShortInt& operator=(const ShortInt& other);
    ShortInt& operator=(ShortInt&& other) noexcept;
    
    // Encrypt a plaintext value
    static ShortInt Encrypt(
        BinFHEContext& cc,
        const ShortIntParams& params,
        uint64_t value,
        const LWEPrivateKey& sk
    );
    
    // Decrypt to plaintext
    uint64_t Decrypt(const LWEPrivateKey& sk) const;
    
    // Get underlying ciphertext
    const LWECiphertext& GetCiphertext() const;
    LWECiphertext& GetCiphertext();
    
    // Get parameters
    const ShortIntParams& GetParams() const;
    
    // Check if this ShortInt has accumulated carries
    bool HasCarry() const;
    
    // Bootstrap to refresh noise and clear carries
    void Bootstrap();
    
    // Get noise budget estimate (for debugging)
    double EstimateNoiseBudget() const;
    
private:
    BinFHEContext* cc_ = nullptr;
    ShortIntParams params_;
    LWECiphertext ct_;
    bool has_carry_ = false;

    // Friend declarations for free functions that need private access
    friend std::pair<ShortInt, ShortInt> Add(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts);
    friend std::pair<ShortInt, ShortInt> AddWithCarry(const ShortInt& a, const ShortInt& b, const ShortInt& carry_in, const ShortIntLUTs& luts);
    friend std::pair<ShortInt, ShortInt> Sub(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts);
    friend std::pair<ShortInt, ShortInt> SubWithBorrow(const ShortInt& a, const ShortInt& b, const ShortInt& borrow_in, const ShortIntLUTs& luts);
    friend std::pair<ShortInt, ShortInt> Mul(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts);
    friend ShortInt Lt(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts);
    friend ShortInt Le(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts);
    friend ShortInt Gt(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts);
    friend ShortInt Ge(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts);
    friend ShortInt Eq(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts);
    friend ShortInt Ne(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts);
    friend ShortInt And(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts);
    friend ShortInt Or(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts);
    friend ShortInt Xor(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts);
    friend ShortInt Not(const ShortInt& a, const ShortIntLUTs& luts);
    friend ShortInt Select(const ShortInt& sel, const ShortInt& if_true, const ShortInt& if_false);
    friend void BatchAdd(const std::vector<ShortInt>& a, const std::vector<ShortInt>& b, std::vector<ShortInt>& sums, std::vector<ShortInt>& carries, const ShortIntLUTs& luts);
    friend void BatchMul(const std::vector<ShortInt>& a, const std::vector<ShortInt>& b, std::vector<ShortInt>& lows, std::vector<ShortInt>& highs, const ShortIntLUTs& luts);
    friend void BatchBootstrap(std::vector<ShortInt>& values);
};

// ============================================================================
// ShortInt Operations
// ============================================================================

/**
 * @brief Add two ShortInts
 * @return (sum, carry) pair
 */
std::pair<ShortInt, ShortInt> Add(
    const ShortInt& a,
    const ShortInt& b,
    const ShortIntLUTs& luts
);

/**
 * @brief Add with input carry
 * @return (sum, carry_out) pair
 */
std::pair<ShortInt, ShortInt> AddWithCarry(
    const ShortInt& a,
    const ShortInt& b,
    const ShortInt& carry_in,
    const ShortIntLUTs& luts
);

/**
 * @brief Subtract two ShortInts
 * @return (difference, borrow) pair
 */
std::pair<ShortInt, ShortInt> Sub(
    const ShortInt& a,
    const ShortInt& b,
    const ShortIntLUTs& luts
);

/**
 * @brief Subtract with borrow
 * @return (difference, borrow_out) pair
 */
std::pair<ShortInt, ShortInt> SubWithBorrow(
    const ShortInt& a,
    const ShortInt& b,
    const ShortInt& borrow_in,
    const ShortIntLUTs& luts
);

/**
 * @brief Multiply two ShortInts
 * @return (low_bits, high_bits) pair
 */
std::pair<ShortInt, ShortInt> Mul(
    const ShortInt& a,
    const ShortInt& b,
    const ShortIntLUTs& luts
);

/**
 * @brief Compare two ShortInts
 * @return encrypted boolean result
 */
ShortInt Lt(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts);
ShortInt Le(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts);
ShortInt Gt(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts);
ShortInt Ge(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts);
ShortInt Eq(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts);
ShortInt Ne(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts);

/**
 * @brief Bitwise operations
 */
ShortInt And(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts);
ShortInt Or(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts);
ShortInt Xor(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts);
ShortInt Not(const ShortInt& a, const ShortIntLUTs& luts);

/**
 * @brief Conditional select
 * @param sel Selector (0 or 1)
 * @param if_true Value if sel == 1
 * @param if_false Value if sel == 0
 */
ShortInt Select(
    const ShortInt& sel,
    const ShortInt& if_true,
    const ShortInt& if_false
);

// ============================================================================
// Batch ShortInt Operations (for GPU throughput)
// ============================================================================

/**
 * @brief Batch add with carry propagation
 * 
 * Adds pairs of ShortInts, returning sums and carries.
 * Optimized for GPU execution via batched bootstrapping.
 */
void BatchAdd(
    const std::vector<ShortInt>& a,
    const std::vector<ShortInt>& b,
    std::vector<ShortInt>& sums,
    std::vector<ShortInt>& carries,
    const ShortIntLUTs& luts
);

/**
 * @brief Batch multiply
 * 
 * Multiplies pairs of ShortInts, returning low and high parts.
 */
void BatchMul(
    const std::vector<ShortInt>& a,
    const std::vector<ShortInt>& b,
    std::vector<ShortInt>& lows,
    std::vector<ShortInt>& highs,
    const ShortIntLUTs& luts
);

/**
 * @brief Batch bootstrap
 * 
 * Refresh noise for multiple ShortInts.
 */
void BatchBootstrap(std::vector<ShortInt>& values);

} // namespace radix
} // namespace lbcrypto

#endif // RADIX_SHORTINT_H
