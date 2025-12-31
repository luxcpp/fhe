// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// RadixInt - Arbitrary-precision encrypted integers using radix representation
// Built on ShortInt limbs with carry propagation.

#ifndef RADIX_RADIX_H
#define RADIX_RADIX_H

#include "radix/shortint.h"
#include <cstdint>
#include <memory>
#include <vector>
#include <string>

namespace lbcrypto {
namespace radix {

// Radix integer parameters
struct RadixParams {
    ShortIntParams limb_params;  // Parameters for each limb
    uint32_t num_limbs;          // Number of limbs
    
    // Total bits of precision
    uint32_t total_bits() const { 
        return limb_params.message_bits * num_limbs; 
    }
    
    // Maximum representable value
    // Note: For unsigned integers only
    uint64_t max_value() const {
        if (total_bits() >= 64) return UINT64_MAX;
        return (1ULL << total_bits()) - 1;
    }
};

// Standard parameter sets for fhEVM types
namespace params {
    // euint8: 8-bit encrypted unsigned integer
    // 4 limbs × 2 bits each = 8 bits
    const RadixParams EUINT8 = {
        .limb_params = { .message_bits = 2, .carry_bits = 2 },
        .num_limbs = 4
    };
    
    // euint16: 16-bit encrypted unsigned integer
    // 8 limbs × 2 bits each = 16 bits
    const RadixParams EUINT16 = {
        .limb_params = { .message_bits = 2, .carry_bits = 2 },
        .num_limbs = 8
    };
    
    // euint32: 32-bit encrypted unsigned integer
    // 16 limbs × 2 bits each = 32 bits
    const RadixParams EUINT32 = {
        .limb_params = { .message_bits = 2, .carry_bits = 2 },
        .num_limbs = 16
    };
    
    // euint64: 64-bit encrypted unsigned integer
    // 32 limbs × 2 bits each = 64 bits
    const RadixParams EUINT64 = {
        .limb_params = { .message_bits = 2, .carry_bits = 2 },
        .num_limbs = 32
    };
    
    // euint128: 128-bit encrypted unsigned integer
    // 64 limbs × 2 bits each = 128 bits
    const RadixParams EUINT128 = {
        .limb_params = { .message_bits = 2, .carry_bits = 2 },
        .num_limbs = 64
    };
    
    // euint256: 256-bit encrypted unsigned integer
    // 128 limbs × 2 bits each = 256 bits (for EVM uint256)
    const RadixParams EUINT256 = {
        .limb_params = { .message_bits = 2, .carry_bits = 2 },
        .num_limbs = 128
    };
    
    // eaddress: 160-bit encrypted address
    // 80 limbs × 2 bits each = 160 bits
    const RadixParams EADDRESS = {
        .limb_params = { .message_bits = 2, .carry_bits = 2 },
        .num_limbs = 80
    };
}

/**
 * @brief RadixInt - Encrypted arbitrary-precision integer
 * 
 * Represents an encrypted integer as a sequence of ShortInt limbs in
 * little-endian order (limbs[0] is least significant).
 * 
 * Arithmetic operations are performed limb-by-limb with carry propagation.
 * Carries are lazily propagated - multiple additions can be performed
 * before a full carry propagation is needed.
 */
class RadixInt {
public:
    RadixInt() = default;
    RadixInt(BinFHEContext& cc, const RadixParams& params);
    ~RadixInt();
    
    // Copy/move
    RadixInt(const RadixInt& other);
    RadixInt(RadixInt&& other) noexcept;
    RadixInt& operator=(const RadixInt& other);
    RadixInt& operator=(RadixInt&& other) noexcept;
    
    // ========================================================================
    // Encryption/Decryption
    // ========================================================================
    
    /**
     * @brief Encrypt a plaintext value
     */
    static RadixInt Encrypt(
        BinFHEContext& cc,
        const RadixParams& params,
        uint64_t value,
        const LWEPrivateKey& sk
    );
    
    /**
     * @brief Encrypt from bytes (big-endian)
     */
    static RadixInt EncryptBytes(
        BinFHEContext& cc,
        const RadixParams& params,
        const std::vector<uint8_t>& bytes,
        const LWEPrivateKey& sk
    );
    
    /**
     * @brief Encrypt a zero
     */
    static RadixInt Zero(BinFHEContext& cc, const RadixParams& params);
    
    /**
     * @brief Decrypt to plaintext
     */
    uint64_t Decrypt(const LWEPrivateKey& sk) const;
    
    /**
     * @brief Decrypt to bytes (big-endian)
     */
    std::vector<uint8_t> DecryptBytes(const LWEPrivateKey& sk) const;
    
    // ========================================================================
    // Accessors
    // ========================================================================
    
    const RadixParams& GetParams() const { return params_; }
    size_t NumLimbs() const { return limbs_.size(); }
    
    const ShortInt& GetLimb(size_t i) const { return limbs_[i]; }
    ShortInt& GetLimb(size_t i) { return limbs_[i]; }
    
    const std::vector<ShortInt>& GetLimbs() const { return limbs_; }
    std::vector<ShortInt>& GetLimbs() { return limbs_; }
    
    // ========================================================================
    // Carry Management
    // ========================================================================
    
    /**
     * @brief Check if any limb has pending carries
     */
    bool HasPendingCarries() const;
    
    /**
     * @brief Propagate all carries through the limbs
     * 
     * This is the expensive operation that requires bootstrapping.
     * Call sparingly - the system tracks noise and auto-propagates when needed.
     */
    void PropagateCarries(const ShortIntLUTs& luts);
    
    /**
     * @brief Estimate total noise level
     */
    double EstimateNoise() const;
    
    // ========================================================================
    // In-place Operations (modify this)
    // ========================================================================
    
    /**
     * @brief Add another RadixInt to this
     */
    void AddInPlace(const RadixInt& other, const ShortIntLUTs& luts);
    
    /**
     * @brief Add a plaintext constant
     */
    void AddScalarInPlace(uint64_t scalar, const ShortIntLUTs& luts);
    
    /**
     * @brief Subtract another RadixInt from this
     */
    void SubInPlace(const RadixInt& other, const ShortIntLUTs& luts);
    
    /**
     * @brief Subtract a plaintext constant
     */
    void SubScalarInPlace(uint64_t scalar, const ShortIntLUTs& luts);
    
    /**
     * @brief Multiply by a plaintext constant
     */
    void MulScalarInPlace(uint64_t scalar, const ShortIntLUTs& luts);
    
    /**
     * @brief Negate this value (two's complement)
     */
    void NegateInPlace(const ShortIntLUTs& luts);
    
    /**
     * @brief Bitwise NOT
     */
    void BitwiseNotInPlace(const ShortIntLUTs& luts);
    
    /**
     * @brief Left shift by constant bits
     */
    void ShlInPlace(uint32_t bits, const ShortIntLUTs& luts);
    
    /**
     * @brief Right shift by constant bits
     */
    void ShrInPlace(uint32_t bits, const ShortIntLUTs& luts);

private:
    BinFHEContext* cc_ = nullptr;
    RadixParams params_;
    std::vector<ShortInt> limbs_;

    // Friend declarations for free functions that need private access
    friend RadixInt Add(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);
    friend RadixInt Sub(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);
    friend RadixInt Mul(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);
    friend RadixInt MulFull(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);
    friend std::pair<RadixInt, RadixInt> Div(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);
    friend RadixInt Mod(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);
    friend RadixInt Lt(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);
    friend RadixInt Le(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);
    friend RadixInt Gt(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);
    friend RadixInt Ge(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);
    friend RadixInt Eq(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);
    friend RadixInt Ne(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);
    friend RadixInt Min(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);
    friend RadixInt Max(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);
    friend RadixInt BitwiseAnd(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);
    friend RadixInt BitwiseOr(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);
    friend RadixInt BitwiseXor(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);
    friend RadixInt BitwiseNot(const RadixInt& a, const ShortIntLUTs& luts);
    friend RadixInt Shl(const RadixInt& a, const RadixInt& bits, const ShortIntLUTs& luts);
    friend RadixInt Shr(const RadixInt& a, const RadixInt& bits, const ShortIntLUTs& luts);
    friend RadixInt Rotl(const RadixInt& a, const RadixInt& bits, const ShortIntLUTs& luts);
    friend RadixInt Rotr(const RadixInt& a, const RadixInt& bits, const ShortIntLUTs& luts);
    friend RadixInt Select(const RadixInt& sel, const RadixInt& if_true, const RadixInt& if_false, const ShortIntLUTs& luts);
    friend void SelectAssign(RadixInt& target, const RadixInt& sel, const RadixInt& value, const ShortIntLUTs& luts);
    friend RadixInt Cast(const RadixInt& a, const RadixParams& new_params);
    friend RadixInt IsZero(const RadixInt& a, const ShortIntLUTs& luts);
    friend RadixInt IsNonZero(const RadixInt& a, const ShortIntLUTs& luts);
};

// ============================================================================
// RadixInt Free Functions
// ============================================================================

/**
 * @brief Add two RadixInts
 */
RadixInt Add(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);

/**
 * @brief Subtract two RadixInts
 */
RadixInt Sub(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);

/**
 * @brief Multiply two RadixInts
 * 
 * Uses schoolbook multiplication with lazy carry propagation.
 * Result has same width as inputs (truncated).
 */
RadixInt Mul(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);

/**
 * @brief Multiply two RadixInts (full width result)
 * 
 * Returns a RadixInt with double the limbs to hold full product.
 */
RadixInt MulFull(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);

/**
 * @brief Divide two RadixInts
 * @return (quotient, remainder) pair
 */
std::pair<RadixInt, RadixInt> Div(
    const RadixInt& a, 
    const RadixInt& b, 
    const ShortIntLUTs& luts
);

/**
 * @brief Modulo operation
 */
RadixInt Mod(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);

// ============================================================================
// Comparison Operations
// ============================================================================

/**
 * @brief Compare two RadixInts
 * @return Encrypted boolean (as RadixInt with 1 limb)
 */
RadixInt Lt(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);
RadixInt Le(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);
RadixInt Gt(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);
RadixInt Ge(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);
RadixInt Eq(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);
RadixInt Ne(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);

/**
 * @brief Return minimum of two RadixInts
 */
RadixInt Min(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);

/**
 * @brief Return maximum of two RadixInts
 */
RadixInt Max(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);

// ============================================================================
// Bitwise Operations
// ============================================================================

RadixInt BitwiseAnd(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);
RadixInt BitwiseOr(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);
RadixInt BitwiseXor(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts);
RadixInt BitwiseNot(const RadixInt& a, const ShortIntLUTs& luts);

/**
 * @brief Shift left by encrypted amount
 */
RadixInt Shl(const RadixInt& a, const RadixInt& bits, const ShortIntLUTs& luts);

/**
 * @brief Shift right by encrypted amount
 */
RadixInt Shr(const RadixInt& a, const RadixInt& bits, const ShortIntLUTs& luts);

/**
 * @brief Rotate left by encrypted amount
 */
RadixInt Rotl(const RadixInt& a, const RadixInt& bits, const ShortIntLUTs& luts);

/**
 * @brief Rotate right by encrypted amount
 */
RadixInt Rotr(const RadixInt& a, const RadixInt& bits, const ShortIntLUTs& luts);

// ============================================================================
// Conditional Operations
// ============================================================================

/**
 * @brief Conditional select
 * @param sel Encrypted boolean selector
 * @param if_true Value if sel == 1
 * @param if_false Value if sel == 0
 */
RadixInt Select(
    const RadixInt& sel,
    const RadixInt& if_true,
    const RadixInt& if_false,
    const ShortIntLUTs& luts
);

/**
 * @brief Conditional assign
 * 
 * If sel == 1, set target = value, else target unchanged.
 */
void SelectAssign(
    RadixInt& target,
    const RadixInt& sel,
    const RadixInt& value,
    const ShortIntLUTs& luts
);

// ============================================================================
// Type Conversion
// ============================================================================

/**
 * @brief Cast to a different width
 * 
 * Truncates if smaller, zero-extends if larger.
 */
RadixInt Cast(const RadixInt& a, const RadixParams& new_params);

/**
 * @brief Convert to/from boolean
 */
RadixInt IsNonZero(const RadixInt& a, const ShortIntLUTs& luts);
RadixInt IsZero(const RadixInt& a, const ShortIntLUTs& luts);

// ============================================================================
// Batch Operations (GPU-optimized)
// ============================================================================

/**
 * @brief Batch add multiple RadixInt pairs
 */
void BatchAdd(
    const std::vector<RadixInt>& a,
    const std::vector<RadixInt>& b,
    std::vector<RadixInt>& results,
    const ShortIntLUTs& luts
);

/**
 * @brief Batch multiply multiple RadixInt pairs
 */
void BatchMul(
    const std::vector<RadixInt>& a,
    const std::vector<RadixInt>& b,
    std::vector<RadixInt>& results,
    const ShortIntLUTs& luts
);

/**
 * @brief Batch carry propagation
 */
void BatchPropagateCarries(
    std::vector<RadixInt>& values,
    const ShortIntLUTs& luts
);

} // namespace radix
} // namespace lbcrypto

#endif // RADIX_RADIX_H
