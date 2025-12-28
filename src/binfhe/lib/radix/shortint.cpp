// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// ShortInt implementation - encrypted small integers (limbs)

#include "radix/shortint.h"
#include "batch/binfhe-batch.h"
#include <stdexcept>
#include <cmath>

namespace lbcrypto {
namespace radix {

// ============================================================================
// ShortIntLUTs Implementation
// ============================================================================

struct ShortIntLUTs::Impl {
    ShortIntParams params;
    
    // Precomputed LUTs
    std::vector<NativeInteger> add_lut;
    std::vector<NativeInteger> add_carry_lut;
    std::vector<NativeInteger> sub_lut;
    std::vector<NativeInteger> sub_borrow_lut;
    std::vector<NativeInteger> mul_lut;
    std::vector<NativeInteger> mul_high_lut;
    std::vector<NativeInteger> lt_lut;
    std::vector<NativeInteger> le_lut;
    std::vector<NativeInteger> gt_lut;
    std::vector<NativeInteger> ge_lut;
    std::vector<NativeInteger> eq_lut;
    std::vector<NativeInteger> ne_lut;
    std::vector<NativeInteger> and_lut;
    std::vector<NativeInteger> or_lut;
    std::vector<NativeInteger> xor_lut;
    std::vector<NativeInteger> not_lut;
    std::vector<NativeInteger> identity_lut;
    std::vector<NativeInteger> clean_carry_lut;
    
    void BuildLUTs() {
        uint64_t p = 1ULL << params.message_bits;
        uint64_t total = 1ULL << params.total_bits();
        
        // For two-input operations, we need p^2 entries
        size_t two_input_size = p * p;
        
        // Add LUT: (a + b) mod p
        add_lut.resize(two_input_size);
        add_carry_lut.resize(two_input_size);
        for (uint64_t a = 0; a < p; ++a) {
            for (uint64_t b = 0; b < p; ++b) {
                uint64_t sum = a + b;
                add_lut[a * p + b] = NativeInteger(sum % p);
                add_carry_lut[a * p + b] = NativeInteger(sum >= p ? 1 : 0);
            }
        }
        
        // Sub LUT: (a - b) mod p
        sub_lut.resize(two_input_size);
        sub_borrow_lut.resize(two_input_size);
        for (uint64_t a = 0; a < p; ++a) {
            for (uint64_t b = 0; b < p; ++b) {
                int64_t diff = static_cast<int64_t>(a) - static_cast<int64_t>(b);
                if (diff < 0) {
                    sub_lut[a * p + b] = NativeInteger(diff + p);
                    sub_borrow_lut[a * p + b] = NativeInteger(1);
                } else {
                    sub_lut[a * p + b] = NativeInteger(diff);
                    sub_borrow_lut[a * p + b] = NativeInteger(0);
                }
            }
        }
        
        // Mul LUT: (a * b) low bits
        mul_lut.resize(two_input_size);
        mul_high_lut.resize(two_input_size);
        for (uint64_t a = 0; a < p; ++a) {
            for (uint64_t b = 0; b < p; ++b) {
                uint64_t prod = a * b;
                mul_lut[a * p + b] = NativeInteger(prod % p);
                mul_high_lut[a * p + b] = NativeInteger(prod / p);
            }
        }
        
        // Comparison LUTs
        lt_lut.resize(two_input_size);
        le_lut.resize(two_input_size);
        gt_lut.resize(two_input_size);
        ge_lut.resize(two_input_size);
        eq_lut.resize(two_input_size);
        ne_lut.resize(two_input_size);
        for (uint64_t a = 0; a < p; ++a) {
            for (uint64_t b = 0; b < p; ++b) {
                lt_lut[a * p + b] = NativeInteger(a < b ? 1 : 0);
                le_lut[a * p + b] = NativeInteger(a <= b ? 1 : 0);
                gt_lut[a * p + b] = NativeInteger(a > b ? 1 : 0);
                ge_lut[a * p + b] = NativeInteger(a >= b ? 1 : 0);
                eq_lut[a * p + b] = NativeInteger(a == b ? 1 : 0);
                ne_lut[a * p + b] = NativeInteger(a != b ? 1 : 0);
            }
        }
        
        // Bitwise LUTs
        and_lut.resize(two_input_size);
        or_lut.resize(two_input_size);
        xor_lut.resize(two_input_size);
        for (uint64_t a = 0; a < p; ++a) {
            for (uint64_t b = 0; b < p; ++b) {
                and_lut[a * p + b] = NativeInteger(a & b);
                or_lut[a * p + b] = NativeInteger(a | b);
                xor_lut[a * p + b] = NativeInteger(a ^ b);
            }
        }
        
        // Single-input LUTs
        not_lut.resize(p);
        identity_lut.resize(p);
        for (uint64_t a = 0; a < p; ++a) {
            not_lut[a] = NativeInteger((~a) & (p - 1));
            identity_lut[a] = NativeInteger(a);
        }
        
        // Clean carry LUT: extract value and clear carry bits
        clean_carry_lut.resize(total);
        for (uint64_t a = 0; a < total; ++a) {
            clean_carry_lut[a] = NativeInteger(a % p);
        }
    }
};

ShortIntLUTs::ShortIntLUTs(const ShortIntParams& params) 
    : impl_(std::make_unique<Impl>()) {
    impl_->params = params;
    impl_->BuildLUTs();
}

ShortIntLUTs::~ShortIntLUTs() = default;

const std::vector<NativeInteger>& ShortIntLUTs::AddLUT() const { return impl_->add_lut; }
const std::vector<NativeInteger>& ShortIntLUTs::AddCarryLUT() const { return impl_->add_carry_lut; }
const std::vector<NativeInteger>& ShortIntLUTs::SubLUT() const { return impl_->sub_lut; }
const std::vector<NativeInteger>& ShortIntLUTs::SubBorrowLUT() const { return impl_->sub_borrow_lut; }
const std::vector<NativeInteger>& ShortIntLUTs::MulLUT() const { return impl_->mul_lut; }
const std::vector<NativeInteger>& ShortIntLUTs::MulHighLUT() const { return impl_->mul_high_lut; }
const std::vector<NativeInteger>& ShortIntLUTs::LtLUT() const { return impl_->lt_lut; }
const std::vector<NativeInteger>& ShortIntLUTs::LeLUT() const { return impl_->le_lut; }
const std::vector<NativeInteger>& ShortIntLUTs::GtLUT() const { return impl_->gt_lut; }
const std::vector<NativeInteger>& ShortIntLUTs::GeLUT() const { return impl_->ge_lut; }
const std::vector<NativeInteger>& ShortIntLUTs::EqLUT() const { return impl_->eq_lut; }
const std::vector<NativeInteger>& ShortIntLUTs::NeLUT() const { return impl_->ne_lut; }
const std::vector<NativeInteger>& ShortIntLUTs::AndLUT() const { return impl_->and_lut; }
const std::vector<NativeInteger>& ShortIntLUTs::OrLUT() const { return impl_->or_lut; }
const std::vector<NativeInteger>& ShortIntLUTs::XorLUT() const { return impl_->xor_lut; }
const std::vector<NativeInteger>& ShortIntLUTs::NotLUT() const { return impl_->not_lut; }
const std::vector<NativeInteger>& ShortIntLUTs::IdentityLUT() const { return impl_->identity_lut; }
const std::vector<NativeInteger>& ShortIntLUTs::CleanCarryLUT() const { return impl_->clean_carry_lut; }

// ============================================================================
// ShortInt Implementation
// ============================================================================

ShortInt::ShortInt(BinFHEContext& cc, const ShortIntParams& params)
    : cc_(&cc), params_(params) {}

ShortInt::~ShortInt() = default;

ShortInt::ShortInt(const ShortInt& other) = default;
ShortInt::ShortInt(ShortInt&& other) noexcept = default;
ShortInt& ShortInt::operator=(const ShortInt& other) = default;
ShortInt& ShortInt::operator=(ShortInt&& other) noexcept = default;

ShortInt ShortInt::Encrypt(
    BinFHEContext& cc,
    const ShortIntParams& params,
    uint64_t value,
    const LWEPrivateKey& sk
) {
    ShortInt result(cc, params);
    
    // Ensure value is within range
    value = value % (1ULL << params.message_bits);
    
    // Encrypt using BinFHE context
    // The value is encoded in the message space
    result.ct_ = cc.Encrypt(sk, value, FRESH, params.total_bits());
    result.has_carry_ = false;
    
    return result;
}

uint64_t ShortInt::Decrypt(const LWEPrivateKey& sk) const {
    if (!cc_) {
        throw std::runtime_error("ShortInt not initialized");
    }
    
    LWEPlaintext pt;
    cc_->Decrypt(sk, ct_, &pt, params_.total_bits());
    
    // Extract message, ignoring carry bits
    return pt % (1ULL << params_.message_bits);
}

const LWECiphertext& ShortInt::GetCiphertext() const { return ct_; }
LWECiphertext& ShortInt::GetCiphertext() { return ct_; }
const ShortIntParams& ShortInt::GetParams() const { return params_; }
bool ShortInt::HasCarry() const { return has_carry_; }

void ShortInt::Bootstrap() {
    if (!cc_) {
        throw std::runtime_error("ShortInt not initialized");
    }
    
    // Use the identity LUT to bootstrap and clear carries
    ShortIntLUTs luts(params_);
    ct_ = cc_->EvalFunc(ct_, luts.CleanCarryLUT());
    has_carry_ = false;
}

double ShortInt::EstimateNoiseBudget() const {
    // Placeholder - would need to track noise through operations
    return 0.0;
}

// ============================================================================
// ShortInt Operations
// ============================================================================

std::pair<ShortInt, ShortInt> Add(
    const ShortInt& a,
    const ShortInt& b,
    const ShortIntLUTs& luts
) {
    if (!a.cc_ || a.cc_ != b.cc_) {
        throw std::runtime_error("ShortInt contexts must match");
    }
    
    ShortInt sum(*a.cc_, a.params_);
    ShortInt carry(*a.cc_, a.params_);
    
    // Pack inputs for two-input LUT evaluation
    // This requires computing (a * p + b) homomorphically
    // Then evaluating the two-output LUT
    
    // For now, use EvalFuncMultiOutputBatch
    std::vector<LWECiphertext> inputs = {/* packed input */};
    std::vector<std::vector<NativeInteger>> lut_pair = {luts.AddLUT(), luts.AddCarryLUT()};
    std::vector<LWECiphertext> outputs;
    
    // EvalFuncMultiOutputBatch(*a.cc_, inputs, lut_pair, outputs, BATCH_DEFAULT);
    
    // sum.ct_ = outputs[0];
    // carry.ct_ = outputs[1];
    
    return {sum, carry};
}

std::pair<ShortInt, ShortInt> AddWithCarry(
    const ShortInt& a,
    const ShortInt& b,
    const ShortInt& carry_in,
    const ShortIntLUTs& luts
) {
    // Add a + b first
    auto [partial_sum, carry1] = Add(a, b, luts);
    
    // Add carry_in
    auto [final_sum, carry2] = Add(partial_sum, carry_in, luts);
    
    // Combine carries (OR them together since at most one can be set)
    ShortInt final_carry = Or(carry1, carry2, luts);
    
    return {final_sum, final_carry};
}

std::pair<ShortInt, ShortInt> Sub(
    const ShortInt& a,
    const ShortInt& b,
    const ShortIntLUTs& luts
) {
    ShortInt diff(*a.cc_, a.params_);
    ShortInt borrow(*a.cc_, a.params_);
    
    // Similar to Add, but using sub LUTs
    
    return {diff, borrow};
}

std::pair<ShortInt, ShortInt> SubWithBorrow(
    const ShortInt& a,
    const ShortInt& b,
    const ShortInt& borrow_in,
    const ShortIntLUTs& luts
) {
    auto [partial_diff, borrow1] = Sub(a, b, luts);
    auto [final_diff, borrow2] = Sub(partial_diff, borrow_in, luts);
    ShortInt final_borrow = Or(borrow1, borrow2, luts);
    return {final_diff, final_borrow};
}

std::pair<ShortInt, ShortInt> Mul(
    const ShortInt& a,
    const ShortInt& b,
    const ShortIntLUTs& luts
) {
    ShortInt low(*a.cc_, a.params_);
    ShortInt high(*a.cc_, a.params_);
    
    // Evaluate mul LUTs
    
    return {low, high};
}

// Comparison operations
ShortInt Lt(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts) {
    ShortInt result(*a.cc_, a.params_);
    // Evaluate lt LUT
    return result;
}

ShortInt Le(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts) {
    ShortInt result(*a.cc_, a.params_);
    return result;
}

ShortInt Gt(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts) {
    ShortInt result(*a.cc_, a.params_);
    return result;
}

ShortInt Ge(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts) {
    ShortInt result(*a.cc_, a.params_);
    return result;
}

ShortInt Eq(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts) {
    ShortInt result(*a.cc_, a.params_);
    return result;
}

ShortInt Ne(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts) {
    ShortInt result(*a.cc_, a.params_);
    return result;
}

// Bitwise operations
ShortInt And(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts) {
    ShortInt result(*a.cc_, a.params_);
    return result;
}

ShortInt Or(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts) {
    ShortInt result(*a.cc_, a.params_);
    return result;
}

ShortInt Xor(const ShortInt& a, const ShortInt& b, const ShortIntLUTs& luts) {
    ShortInt result(*a.cc_, a.params_);
    return result;
}

ShortInt Not(const ShortInt& a, const ShortIntLUTs& luts) {
    ShortInt result(*a.cc_, a.params_);
    return result;
}

ShortInt Select(
    const ShortInt& sel,
    const ShortInt& if_true,
    const ShortInt& if_false
) {
    ShortInt result(*sel.cc_, sel.params_);
    // Implement as CMUX
    return result;
}

// ============================================================================
// Batch Operations
// ============================================================================

void BatchAdd(
    const std::vector<ShortInt>& a,
    const std::vector<ShortInt>& b,
    std::vector<ShortInt>& sums,
    std::vector<ShortInt>& carries,
    const ShortIntLUTs& luts
) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Input vectors must have same size");
    }
    
    sums.resize(a.size());
    carries.resize(a.size());
    
    // Collect ciphertexts for batch processing
    std::vector<LWECiphertext> a_cts, b_cts;
    for (size_t i = 0; i < a.size(); ++i) {
        a_cts.push_back(a[i].GetCiphertext());
        b_cts.push_back(b[i].GetCiphertext());
    }
    
    // TODO: Implement batch LUT evaluation
    // This would use EvalFuncMultiOutputBatch
}

void BatchMul(
    const std::vector<ShortInt>& a,
    const std::vector<ShortInt>& b,
    std::vector<ShortInt>& lows,
    std::vector<ShortInt>& highs,
    const ShortIntLUTs& luts
) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Input vectors must have same size");
    }
    
    lows.resize(a.size());
    highs.resize(a.size());
    
    // TODO: Implement batch multiplication
}

void BatchBootstrap(std::vector<ShortInt>& values) {
    if (values.empty()) return;
    
    // Collect ciphertexts
    std::vector<LWECiphertext> cts;
    for (auto& v : values) {
        cts.push_back(v.GetCiphertext());
    }
    
    // Batch bootstrap
    std::vector<LWECiphertext> out_cts;
    BootstrapBatch(*values[0].cc_, cts, out_cts, BATCH_DEFAULT);
    
    // Update values
    for (size_t i = 0; i < values.size(); ++i) {
        values[i].GetCiphertext() = out_cts[i];
        values[i].has_carry_ = false;
    }
}

} // namespace radix
} // namespace lbcrypto
