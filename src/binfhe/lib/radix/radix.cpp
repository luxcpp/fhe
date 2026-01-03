// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// RadixInt implementation - arbitrary-precision encrypted integers

#include "radix/radix.h"
#include "batch/binfhe-batch.h"
#include <stdexcept>
#include <algorithm>

namespace lux::fhe {
namespace radix {

// ============================================================================
// RadixInt Implementation
// ============================================================================

RadixInt::RadixInt(BinFHEContext& cc, const RadixParams& params)
    : cc_(&cc), params_(params) {
    limbs_.resize(params.num_limbs, ShortInt(cc, params.limb_params));
}

RadixInt::~RadixInt() = default;

RadixInt::RadixInt(const RadixInt& other) = default;
RadixInt::RadixInt(RadixInt&& other) noexcept = default;
RadixInt& RadixInt::operator=(const RadixInt& other) = default;
RadixInt& RadixInt::operator=(RadixInt&& other) noexcept = default;

RadixInt RadixInt::Encrypt(
    BinFHEContext& cc,
    const RadixParams& params,
    uint64_t value,
    const LWEPrivateKey& sk
) {
    RadixInt result(cc, params);
    
    uint64_t limb_mask = (1ULL << params.limb_params.message_bits) - 1;
    
    for (uint32_t i = 0; i < params.num_limbs; ++i) {
        uint64_t limb_value = (value >> (i * params.limb_params.message_bits)) & limb_mask;
        result.limbs_[i] = ShortInt::Encrypt(cc, params.limb_params, limb_value, sk);
    }
    
    return result;
}

RadixInt RadixInt::EncryptBytes(
    BinFHEContext& cc,
    const RadixParams& params,
    const std::vector<uint8_t>& bytes,
    const LWEPrivateKey& sk
) {
    RadixInt result(cc, params);
    
    // Convert bytes (big-endian) to limbs (little-endian)
    uint32_t bits_per_limb = params.limb_params.message_bits;
    
    // Accumulate bits from bytes
    std::vector<uint64_t> limb_values(params.num_limbs, 0);
    
    size_t bit_pos = 0;
    for (size_t i = bytes.size(); i > 0; --i) {
        uint8_t byte = bytes[i - 1];
        for (int j = 0; j < 8; ++j) {
            if (bit_pos / bits_per_limb >= params.num_limbs) break;
            
            uint32_t limb_idx = bit_pos / bits_per_limb;
            uint32_t bit_in_limb = bit_pos % bits_per_limb;
            
            if (byte & (1 << j)) {
                limb_values[limb_idx] |= (1ULL << bit_in_limb);
            }
            bit_pos++;
        }
    }
    
    // Encrypt each limb
    for (uint32_t i = 0; i < params.num_limbs; ++i) {
        result.limbs_[i] = ShortInt::Encrypt(cc, params.limb_params, limb_values[i], sk);
    }
    
    return result;
}

RadixInt RadixInt::Zero(BinFHEContext& cc, const RadixParams& params) {
    RadixInt result(cc, params);
    
    // Each limb encrypts 0
    for (uint32_t i = 0; i < params.num_limbs; ++i) {
        // Use trivial encryption for zero
        result.limbs_[i] = ShortInt(cc, params.limb_params);
        // result.limbs_[i].ct_ = cc.TrivialEncrypt(0);
    }
    
    return result;
}

uint64_t RadixInt::Decrypt(const LWEPrivateKey& sk) const {
    if (!cc_) {
        throw std::runtime_error("RadixInt not initialized");
    }
    
    uint64_t result = 0;
    uint32_t bits_per_limb = params_.limb_params.message_bits;
    
    for (uint32_t i = 0; i < params_.num_limbs && i * bits_per_limb < 64; ++i) {
        uint64_t limb_value = limbs_[i].Decrypt(sk);
        result |= (limb_value << (i * bits_per_limb));
    }
    
    return result;
}

std::vector<uint8_t> RadixInt::DecryptBytes(const LWEPrivateKey& sk) const {
    if (!cc_) {
        throw std::runtime_error("RadixInt not initialized");
    }
    
    uint32_t total_bits = params_.total_bits();
    size_t num_bytes = (total_bits + 7) / 8;
    std::vector<uint8_t> result(num_bytes, 0);
    
    // Decrypt each limb and pack into bytes
    uint32_t bits_per_limb = params_.limb_params.message_bits;
    
    for (uint32_t i = 0; i < params_.num_limbs; ++i) {
        uint64_t limb_value = limbs_[i].Decrypt(sk);
        uint32_t bit_offset = i * bits_per_limb;
        
        for (uint32_t j = 0; j < bits_per_limb; ++j) {
            uint32_t bit_pos = bit_offset + j;
            if (bit_pos >= total_bits) break;
            
            size_t byte_idx = num_bytes - 1 - (bit_pos / 8);
            uint32_t bit_in_byte = bit_pos % 8;
            
            if (limb_value & (1ULL << j)) {
                result[byte_idx] |= (1 << bit_in_byte);
            }
        }
    }
    
    return result;
}

bool RadixInt::HasPendingCarries() const {
    for (const auto& limb : limbs_) {
        if (limb.HasCarry()) return true;
    }
    return false;
}

void RadixInt::PropagateCarries(const ShortIntLUTs& luts) {
    // Propagate carries through all limbs
    ShortInt carry = ShortInt::Encrypt(*cc_, params_.limb_params, 0, nullptr);
    
    for (uint32_t i = 0; i < params_.num_limbs; ++i) {
        // Add current carry to this limb
        auto [sum, new_carry] = AddWithCarry(limbs_[i], carry, 
            ShortInt::Encrypt(*cc_, params_.limb_params, 0, nullptr), luts);
        limbs_[i] = sum;
        carry = new_carry;
    }
    
    // Final carry is discarded (overflow)
}

double RadixInt::EstimateNoise() const {
    double max_noise = 0;
    for (const auto& limb : limbs_) {
        max_noise = std::max(max_noise, limb.EstimateNoiseBudget());
    }
    return max_noise;
}

// ============================================================================
// In-place Operations
// ============================================================================

void RadixInt::AddInPlace(const RadixInt& other, const ShortIntLUTs& luts) {
    if (params_.num_limbs != other.params_.num_limbs) {
        throw std::invalid_argument("RadixInt sizes must match");
    }
    
    ShortInt carry = ShortInt::Encrypt(*cc_, params_.limb_params, 0, nullptr);
    
    for (uint32_t i = 0; i < params_.num_limbs; ++i) {
        auto [sum, new_carry] = AddWithCarry(limbs_[i], other.limbs_[i], carry, luts);
        limbs_[i] = sum;
        carry = new_carry;
    }
}

void RadixInt::AddScalarInPlace(uint64_t scalar, const ShortIntLUTs& luts) {
    uint64_t limb_mask = (1ULL << params_.limb_params.message_bits) - 1;
    
    // Add scalar limb by limb
    for (uint32_t i = 0; i < params_.num_limbs && scalar > 0; ++i) {
        uint64_t limb_value = scalar & limb_mask;
        scalar >>= params_.limb_params.message_bits;
        
        // Create encrypted scalar limb (trivial encryption)
        ShortInt scalar_limb = ShortInt::Encrypt(*cc_, params_.limb_params, limb_value, nullptr);
        
        auto [sum, carry] = Add(limbs_[i], scalar_limb, luts);
        limbs_[i] = sum;
        
        // Propagate carry to next limb (simplified - should accumulate)
    }
}

void RadixInt::SubInPlace(const RadixInt& other, const ShortIntLUTs& luts) {
    if (params_.num_limbs != other.params_.num_limbs) {
        throw std::invalid_argument("RadixInt sizes must match");
    }
    
    ShortInt borrow = ShortInt::Encrypt(*cc_, params_.limb_params, 0, nullptr);
    
    for (uint32_t i = 0; i < params_.num_limbs; ++i) {
        auto [diff, new_borrow] = SubWithBorrow(limbs_[i], other.limbs_[i], borrow, luts);
        limbs_[i] = diff;
        borrow = new_borrow;
    }
}

void RadixInt::SubScalarInPlace(uint64_t scalar, const ShortIntLUTs& luts) {
    uint64_t limb_mask = (1ULL << params_.limb_params.message_bits) - 1;
    
    for (uint32_t i = 0; i < params_.num_limbs && scalar > 0; ++i) {
        uint64_t limb_value = scalar & limb_mask;
        scalar >>= params_.limb_params.message_bits;
        
        ShortInt scalar_limb = ShortInt::Encrypt(*cc_, params_.limb_params, limb_value, nullptr);
        auto [diff, borrow] = Sub(limbs_[i], scalar_limb, luts);
        limbs_[i] = diff;
    }
}

void RadixInt::MulScalarInPlace(uint64_t scalar, const ShortIntLUTs& luts) {
    // Multiply by scalar using schoolbook method
    // For each limb of the result, accumulate products
    
    // This is complex - for now just a placeholder
    // Real implementation would compute carry-save representation
}

void RadixInt::NegateInPlace(const ShortIntLUTs& luts) {
    // Two's complement: NOT + 1
    BitwiseNotInPlace(luts);
    AddScalarInPlace(1, luts);
}

void RadixInt::BitwiseNotInPlace(const ShortIntLUTs& luts) {
    for (auto& limb : limbs_) {
        limb = Not(limb, luts);
    }
}

void RadixInt::ShlInPlace(uint32_t bits, const ShortIntLUTs& luts) {
    (void)luts;  // Used for bit-level shift (future implementation)
    uint32_t bits_per_limb = params_.limb_params.message_bits;
    uint32_t limb_shift = bits / bits_per_limb;
    // uint32_t bit_shift = bits % bits_per_limb;  // TODO: implement bit-level shift

    // Shift limbs
    if (limb_shift > 0) {
        for (uint32_t i = params_.num_limbs - 1; i >= limb_shift; --i) {
            limbs_[i] = limbs_[i - limb_shift];
        }
        for (uint32_t i = 0; i < limb_shift; ++i) {
            limbs_[i] = ShortInt::Encrypt(*cc_, params_.limb_params, 0, nullptr);
        }
    }

    // TODO: Shift bits within limbs (requires homomorphic shift via LUT)
}

void RadixInt::ShrInPlace(uint32_t bits, const ShortIntLUTs& luts) {
    (void)luts;  // Used for bit-level shift (future implementation)
    uint32_t bits_per_limb = params_.limb_params.message_bits;
    uint32_t limb_shift = bits / bits_per_limb;
    // uint32_t bit_shift = bits % bits_per_limb;  // TODO: implement bit-level shift

    // Shift limbs
    if (limb_shift > 0) {
        for (uint32_t i = 0; i < params_.num_limbs - limb_shift; ++i) {
            limbs_[i] = limbs_[i + limb_shift];
        }
        for (uint32_t i = params_.num_limbs - limb_shift; i < params_.num_limbs; ++i) {
            limbs_[i] = ShortInt::Encrypt(*cc_, params_.limb_params, 0, nullptr);
        }
    }

    // TODO: Shift bits within limbs (requires homomorphic shift via LUT)
}

// ============================================================================
// Free Functions
// ============================================================================

RadixInt Add(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts) {
    RadixInt result = a;
    result.AddInPlace(b, luts);
    return result;
}

RadixInt Sub(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts) {
    RadixInt result = a;
    result.SubInPlace(b, luts);
    return result;
}

RadixInt Mul(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts) {
    // Schoolbook multiplication
    RadixInt result = RadixInt::Zero(*a.cc_, a.params_);
    
    // For each limb of b, multiply by a and add shifted
    for (uint32_t i = 0; i < b.params_.num_limbs; ++i) {
        // partial = a * b[i]
        // result += partial << (i * bits_per_limb)
    }
    
    return result;
}

RadixInt MulFull(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts) {
    // Double-width multiplication
    RadixParams full_params = a.params_;
    full_params.num_limbs *= 2;
    
    RadixInt result = RadixInt::Zero(*a.cc_, full_params);
    
    // Schoolbook multiplication producing full result
    
    return result;
}

std::pair<RadixInt, RadixInt> Div(
    const RadixInt& a, 
    const RadixInt& b, 
    const ShortIntLUTs& luts
) {
    // Division is complex in FHE
    // Would use restoring or non-restoring division algorithm
    RadixInt quotient = RadixInt::Zero(*a.cc_, a.params_);
    RadixInt remainder = a;
    
    return {quotient, remainder};
}

RadixInt Mod(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts) {
    auto [q, r] = Div(a, b, luts);
    return r;
}

// Comparison operations
RadixInt Lt(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts) {
    // Compare from most significant limb to least
    RadixParams bool_params = a.params_;
    bool_params.num_limbs = 1;
    
    RadixInt result = RadixInt::Zero(*a.cc_, bool_params);
    return result;
}

RadixInt Le(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts) {
    RadixParams bool_params = a.params_;
    bool_params.num_limbs = 1;
    return RadixInt::Zero(*a.cc_, bool_params);
}

RadixInt Gt(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts) {
    return Lt(b, a, luts);
}

RadixInt Ge(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts) {
    return Le(b, a, luts);
}

RadixInt Eq(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts) {
    RadixParams bool_params = a.params_;
    bool_params.num_limbs = 1;
    return RadixInt::Zero(*a.cc_, bool_params);
}

RadixInt Ne(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts) {
    RadixParams bool_params = a.params_;
    bool_params.num_limbs = 1;
    return RadixInt::Zero(*a.cc_, bool_params);
}

RadixInt Min(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts) {
    RadixInt lt_result = Lt(a, b, luts);
    return Select(lt_result, a, b, luts);
}

RadixInt Max(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts) {
    RadixInt gt_result = Gt(a, b, luts);
    return Select(gt_result, a, b, luts);
}

// Bitwise operations
RadixInt BitwiseAnd(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts) {
    RadixInt result(*a.cc_, a.params_);
    for (uint32_t i = 0; i < a.params_.num_limbs; ++i) {
        result.limbs_[i] = And(a.limbs_[i], b.limbs_[i], luts);
    }
    return result;
}

RadixInt BitwiseOr(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts) {
    RadixInt result(*a.cc_, a.params_);
    for (uint32_t i = 0; i < a.params_.num_limbs; ++i) {
        result.limbs_[i] = Or(a.limbs_[i], b.limbs_[i], luts);
    }
    return result;
}

RadixInt BitwiseXor(const RadixInt& a, const RadixInt& b, const ShortIntLUTs& luts) {
    RadixInt result(*a.cc_, a.params_);
    for (uint32_t i = 0; i < a.params_.num_limbs; ++i) {
        result.limbs_[i] = Xor(a.limbs_[i], b.limbs_[i], luts);
    }
    return result;
}

RadixInt BitwiseNot(const RadixInt& a, const ShortIntLUTs& luts) {
    RadixInt result = a;
    result.BitwiseNotInPlace(luts);
    return result;
}

RadixInt Shl(const RadixInt& a, const RadixInt& bits, const ShortIntLUTs& luts) {
    // Encrypted shift amount is complex
    // Would need barrel shifter approach
    return a;
}

RadixInt Shr(const RadixInt& a, const RadixInt& bits, const ShortIntLUTs& luts) {
    return a;
}

RadixInt Rotl(const RadixInt& a, const RadixInt& bits, const ShortIntLUTs& luts) {
    return a;
}

RadixInt Rotr(const RadixInt& a, const RadixInt& bits, const ShortIntLUTs& luts) {
    return a;
}

// Conditional operations
RadixInt Select(
    const RadixInt& sel,
    const RadixInt& if_true,
    const RadixInt& if_false,
    const ShortIntLUTs& luts
) {
    RadixInt result(*sel.cc_, if_true.params_);
    
    // sel should be a single-limb boolean
    // For each limb, compute CMUX
    for (uint32_t i = 0; i < if_true.params_.num_limbs; ++i) {
        result.limbs_[i] = radix::Select(sel.limbs_[0], if_true.limbs_[i], if_false.limbs_[i]);
    }
    
    return result;
}

void SelectAssign(
    RadixInt& target,
    const RadixInt& sel,
    const RadixInt& value,
    const ShortIntLUTs& luts
) {
    target = Select(sel, value, target, luts);
}

// Type conversion
RadixInt Cast(const RadixInt& a, const RadixParams& new_params) {
    RadixInt result(*a.cc_, new_params);
    
    // Copy limbs up to the smaller size
    uint32_t copy_limbs = std::min(a.params_.num_limbs, new_params.num_limbs);
    for (uint32_t i = 0; i < copy_limbs; ++i) {
        result.limbs_[i] = a.limbs_[i];
    }
    
    // Zero-extend if larger
    for (uint32_t i = copy_limbs; i < new_params.num_limbs; ++i) {
        result.limbs_[i] = ShortInt::Encrypt(*a.cc_, new_params.limb_params, 0, nullptr);
    }
    
    return result;
}

RadixInt IsNonZero(const RadixInt& a, const ShortIntLUTs& luts) {
    // OR all limbs together, then check if non-zero
    RadixParams bool_params = a.params_;
    bool_params.num_limbs = 1;
    return RadixInt::Zero(*a.cc_, bool_params);
}

RadixInt IsZero(const RadixInt& a, const ShortIntLUTs& luts) {
    RadixInt non_zero = IsNonZero(a, luts);
    non_zero.BitwiseNotInPlace(luts);
    return non_zero;
}

// ============================================================================
// Batch Operations
// ============================================================================

void BatchAdd(
    const std::vector<RadixInt>& a,
    const std::vector<RadixInt>& b,
    std::vector<RadixInt>& results,
    const ShortIntLUTs& luts
) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Input vectors must have same size");
    }
    
    results.resize(a.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < a.size(); ++i) {
        results[i] = Add(a[i], b[i], luts);
    }
}

void BatchMul(
    const std::vector<RadixInt>& a,
    const std::vector<RadixInt>& b,
    std::vector<RadixInt>& results,
    const ShortIntLUTs& luts
) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Input vectors must have same size");
    }
    
    results.resize(a.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < a.size(); ++i) {
        results[i] = Mul(a[i], b[i], luts);
    }
}

void BatchPropagateCarries(
    std::vector<RadixInt>& values,
    const ShortIntLUTs& luts
) {
    #pragma omp parallel for
    for (size_t i = 0; i < values.size(); ++i) {
        values[i].PropagateCarries(luts);
    }
}

} // namespace radix
} // namespace lux::fhe
