// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// fhEVM Operations Implementation

#include "fhevm/fhevm.h"
#include "batch/binfhe-batch.h"
#include "binfhecontext-ser.h"
#include <stdexcept>
#include <random>
#include <algorithm>
#include <sstream>

namespace lbcrypto {
namespace fhevm {

// ============================================================================
// Type Utilities
// ============================================================================

uint32_t FheTypeBits(FheType type) {
    switch (type) {
        case FheType::EBOOL: return 1;
        case FheType::EUINT4: return 4;
        case FheType::EUINT8: return 8;
        case FheType::EUINT16: return 16;
        case FheType::EUINT32: return 32;
        case FheType::EUINT64: return 64;
        case FheType::EUINT128: return 128;
        case FheType::EUINT160: return 160;
        case FheType::EUINT256: return 256;
        default: return 0;
    }
}

std::string FheTypeName(FheType type) {
    switch (type) {
        case FheType::EBOOL: return "ebool";
        case FheType::EUINT4: return "euint4";
        case FheType::EUINT8: return "euint8";
        case FheType::EUINT16: return "euint16";
        case FheType::EUINT32: return "euint32";
        case FheType::EUINT64: return "euint64";
        case FheType::EUINT128: return "euint128";
        case FheType::EUINT160: return "euint160";
        case FheType::EUINT256: return "euint256";
        default: return "unknown";
    }
}

// ============================================================================
// FheContext Implementation
// ============================================================================

FheContext::FheContext(BINFHE_PARAMSET params) {
    cc_ = std::make_unique<BinFHEContext>();
    cc_->GenerateBinFHEContext(params, GINX);
    
    // Create LUT tables for radix operations
    radix::ShortIntParams limb_params{2, 2};  // 2-bit message, 2-bit carry
    luts_ = std::make_unique<radix::ShortIntLUTs>(limb_params);
}

FheContext::~FheContext() = default;

FheContext::FheContext(FheContext&&) noexcept = default;
FheContext& FheContext::operator=(FheContext&&) noexcept = default;

radix::RadixParams FheContext::GetRadixParams(FheType type) const {
    radix::ShortIntParams limb_params{2, 2};  // 2-bit message, 2-bit carry
    
    uint32_t bits = FheTypeBits(type);
    uint32_t num_limbs = (bits + 1) / 2;  // 2 bits per limb
    
    return radix::RadixParams{limb_params, num_limbs};
}

// ============================================================================
// Key Generation
// ============================================================================

LWEPrivateKey FheContext::KeyGen() {
    return cc_->KeyGen();
}

LWEPublicKey FheContext::PublicKeyGen(const LWEPrivateKey& sk) {
    return cc_->KeyGenPublic(sk);
}

void FheContext::BootstrapKeyGen(const LWEPrivateKey& sk) {
    cc_->BTKeyGen(sk);
}

void FheContext::SwitchKeyGen(const LWEPrivateKey& sk) {
    // Key switching key is generated as part of BTKeyGen in OpenFHE
}

bool FheContext::HasBootstrapKey() const {
    return cc_->GetRefreshKey() != nullptr;
}

// ============================================================================
// Encryption / Decryption
// ============================================================================

radix::RadixInt FheContext::Encrypt(
    const LWEPrivateKey& sk,
    uint64_t value,
    FheType type
) {
    auto params = GetRadixParams(type);
    return radix::RadixInt::Encrypt(*cc_, params, value, sk);
}

radix::RadixInt FheContext::EncryptBytes(
    const LWEPrivateKey& sk,
    const std::vector<uint8_t>& bytes,
    FheType type
) {
    auto params = GetRadixParams(type);
    return radix::RadixInt::EncryptBytes(*cc_, params, bytes, sk);
}

radix::RadixInt FheContext::EncryptPublic(
    const LWEPublicKey& pk,
    uint64_t value,
    FheType type
) {
    auto params = GetRadixParams(type);
    radix::RadixInt result(*cc_, params);
    
    uint32_t bits_per_limb = params.limb_params.message_bits;
    uint64_t limb_mask = (1ULL << bits_per_limb) - 1;
    
    for (uint32_t i = 0; i < params.num_limbs; ++i) {
        uint64_t limb_value = (value >> (i * bits_per_limb)) & limb_mask;
        // Use public key encryption
        auto ct = cc_->Encrypt(pk, limb_value, FRESH, params.limb_params.total_bits());
        result.GetLimb(i).GetCiphertext() = ct;
    }
    
    return result;
}

radix::RadixInt FheContext::TrivialEncrypt(uint64_t value, FheType type) {
    auto params = GetRadixParams(type);
    radix::RadixInt result(*cc_, params);
    
    uint32_t bits_per_limb = params.limb_params.message_bits;
    uint64_t limb_mask = (1ULL << bits_per_limb) - 1;
    
    for (uint32_t i = 0; i < params.num_limbs; ++i) {
        uint64_t limb_value = (value >> (i * bits_per_limb)) & limb_mask;
        // Trivial encryption: ciphertext with no secret
        auto ct = cc_->Encrypt(nullptr, limb_value, TRIVIAL, params.limb_params.total_bits());
        result.GetLimb(i).GetCiphertext() = ct;
    }
    
    return result;
}

radix::RadixInt FheContext::TrivialEncryptBytes(
    const std::vector<uint8_t>& bytes,
    FheType type
) {
    // Convert bytes to uint64 and use TrivialEncrypt
    uint64_t value = 0;
    size_t max_bytes = std::min(bytes.size(), size_t(8));
    for (size_t i = 0; i < max_bytes; ++i) {
        value |= static_cast<uint64_t>(bytes[bytes.size() - 1 - i]) << (i * 8);
    }
    return TrivialEncrypt(value, type);
}

uint64_t FheContext::Decrypt(const LWEPrivateKey& sk, const radix::RadixInt& ct) {
    return ct.Decrypt(sk);
}

std::vector<uint8_t> FheContext::DecryptBytes(
    const LWEPrivateKey& sk,
    const radix::RadixInt& ct
) {
    return ct.DecryptBytes(sk);
}

// ============================================================================
// Arithmetic Operations
// ============================================================================

radix::RadixInt FheContext::Add(const radix::RadixInt& a, const radix::RadixInt& b) {
    return radix::Add(a, b, *luts_);
}

radix::RadixInt FheContext::AddScalar(const radix::RadixInt& a, uint64_t b) {
    radix::RadixInt result = a;
    result.AddScalarInPlace(b, *luts_);
    return result;
}

radix::RadixInt FheContext::Sub(const radix::RadixInt& a, const radix::RadixInt& b) {
    return radix::Sub(a, b, *luts_);
}

radix::RadixInt FheContext::SubScalar(const radix::RadixInt& a, uint64_t b) {
    radix::RadixInt result = a;
    result.SubScalarInPlace(b, *luts_);
    return result;
}

radix::RadixInt FheContext::Neg(const radix::RadixInt& a) {
    radix::RadixInt result = a;
    result.NegateInPlace(*luts_);
    return result;
}

radix::RadixInt FheContext::Mul(const radix::RadixInt& a, const radix::RadixInt& b) {
    return radix::Mul(a, b, *luts_);
}

radix::RadixInt FheContext::MulScalar(const radix::RadixInt& a, uint64_t b) {
    radix::RadixInt result = a;
    result.MulScalarInPlace(b, *luts_);
    return result;
}

radix::RadixInt FheContext::Div(const radix::RadixInt& a, const radix::RadixInt& b) {
    auto [quotient, remainder] = radix::Div(a, b, *luts_);
    return quotient;
}

radix::RadixInt FheContext::DivScalar(const radix::RadixInt& a, uint64_t b) {
    auto b_enc = TrivialEncrypt(b, GetType(a));
    return Div(a, b_enc);
}

radix::RadixInt FheContext::Rem(const radix::RadixInt& a, const radix::RadixInt& b) {
    return radix::Mod(a, b, *luts_);
}

radix::RadixInt FheContext::RemScalar(const radix::RadixInt& a, uint64_t b) {
    auto b_enc = TrivialEncrypt(b, GetType(a));
    return Rem(a, b_enc);
}

// ============================================================================
// Comparison Operations
// ============================================================================

radix::RadixInt FheContext::Eq(const radix::RadixInt& a, const radix::RadixInt& b) {
    return radix::Eq(a, b, *luts_);
}

radix::RadixInt FheContext::Ne(const radix::RadixInt& a, const radix::RadixInt& b) {
    return radix::Ne(a, b, *luts_);
}

radix::RadixInt FheContext::Lt(const radix::RadixInt& a, const radix::RadixInt& b) {
    return radix::Lt(a, b, *luts_);
}

radix::RadixInt FheContext::Le(const radix::RadixInt& a, const radix::RadixInt& b) {
    return radix::Le(a, b, *luts_);
}

radix::RadixInt FheContext::Gt(const radix::RadixInt& a, const radix::RadixInt& b) {
    return radix::Gt(a, b, *luts_);
}

radix::RadixInt FheContext::Ge(const radix::RadixInt& a, const radix::RadixInt& b) {
    return radix::Ge(a, b, *luts_);
}

radix::RadixInt FheContext::Min(const radix::RadixInt& a, const radix::RadixInt& b) {
    return radix::Min(a, b, *luts_);
}

radix::RadixInt FheContext::Max(const radix::RadixInt& a, const radix::RadixInt& b) {
    return radix::Max(a, b, *luts_);
}

radix::RadixInt FheContext::EqScalar(const radix::RadixInt& a, uint64_t b) {
    auto b_enc = TrivialEncrypt(b, GetType(a));
    return Eq(a, b_enc);
}

radix::RadixInt FheContext::NeScalar(const radix::RadixInt& a, uint64_t b) {
    auto b_enc = TrivialEncrypt(b, GetType(a));
    return Ne(a, b_enc);
}

radix::RadixInt FheContext::LtScalar(const radix::RadixInt& a, uint64_t b) {
    auto b_enc = TrivialEncrypt(b, GetType(a));
    return Lt(a, b_enc);
}

radix::RadixInt FheContext::LeScalar(const radix::RadixInt& a, uint64_t b) {
    auto b_enc = TrivialEncrypt(b, GetType(a));
    return Le(a, b_enc);
}

radix::RadixInt FheContext::GtScalar(const radix::RadixInt& a, uint64_t b) {
    auto b_enc = TrivialEncrypt(b, GetType(a));
    return Gt(a, b_enc);
}

radix::RadixInt FheContext::GeScalar(const radix::RadixInt& a, uint64_t b) {
    auto b_enc = TrivialEncrypt(b, GetType(a));
    return Ge(a, b_enc);
}

// ============================================================================
// Bitwise Operations
// ============================================================================

radix::RadixInt FheContext::And(const radix::RadixInt& a, const radix::RadixInt& b) {
    return radix::BitwiseAnd(a, b, *luts_);
}

radix::RadixInt FheContext::Or(const radix::RadixInt& a, const radix::RadixInt& b) {
    return radix::BitwiseOr(a, b, *luts_);
}

radix::RadixInt FheContext::Xor(const radix::RadixInt& a, const radix::RadixInt& b) {
    return radix::BitwiseXor(a, b, *luts_);
}

radix::RadixInt FheContext::Not(const radix::RadixInt& a) {
    return radix::BitwiseNot(a, *luts_);
}

radix::RadixInt FheContext::AndScalar(const radix::RadixInt& a, uint64_t b) {
    auto b_enc = TrivialEncrypt(b, GetType(a));
    return And(a, b_enc);
}

radix::RadixInt FheContext::OrScalar(const radix::RadixInt& a, uint64_t b) {
    auto b_enc = TrivialEncrypt(b, GetType(a));
    return Or(a, b_enc);
}

radix::RadixInt FheContext::XorScalar(const radix::RadixInt& a, uint64_t b) {
    auto b_enc = TrivialEncrypt(b, GetType(a));
    return Xor(a, b_enc);
}

// ============================================================================
// Shift and Rotate Operations
// ============================================================================

radix::RadixInt FheContext::Shl(const radix::RadixInt& a, const radix::RadixInt& bits) {
    return radix::Shl(a, bits, *luts_);
}

radix::RadixInt FheContext::ShlScalar(const radix::RadixInt& a, uint32_t bits) {
    radix::RadixInt result = a;
    result.ShlInPlace(bits, *luts_);
    return result;
}

radix::RadixInt FheContext::Shr(const radix::RadixInt& a, const radix::RadixInt& bits) {
    return radix::Shr(a, bits, *luts_);
}

radix::RadixInt FheContext::ShrScalar(const radix::RadixInt& a, uint32_t bits) {
    radix::RadixInt result = a;
    result.ShrInPlace(bits, *luts_);
    return result;
}

radix::RadixInt FheContext::Rotl(const radix::RadixInt& a, const radix::RadixInt& bits) {
    return radix::Rotl(a, bits, *luts_);
}

radix::RadixInt FheContext::RotlScalar(const radix::RadixInt& a, uint32_t bits) {
    uint32_t total_bits = a.GetParams().total_bits();
    bits = bits % total_bits;
    auto high = ShlScalar(a, bits);
    auto low = ShrScalar(a, total_bits - bits);
    return Or(high, low);
}

radix::RadixInt FheContext::Rotr(const radix::RadixInt& a, const radix::RadixInt& bits) {
    return radix::Rotr(a, bits, *luts_);
}

radix::RadixInt FheContext::RotrScalar(const radix::RadixInt& a, uint32_t bits) {
    uint32_t total_bits = a.GetParams().total_bits();
    bits = bits % total_bits;
    auto low = ShrScalar(a, bits);
    auto high = ShlScalar(a, total_bits - bits);
    return Or(high, low);
}

// ============================================================================
// Control Flow Operations
// ============================================================================

radix::RadixInt FheContext::Select(
    const radix::RadixInt& cond,
    const radix::RadixInt& if_true,
    const radix::RadixInt& if_false
) {
    return radix::Select(cond, if_true, if_false, *luts_);
}

radix::RadixInt FheContext::IfThenElse(
    const radix::RadixInt& cond,
    const radix::RadixInt& then_val,
    const radix::RadixInt& else_val
) {
    return Select(cond, then_val, else_val);
}

radix::RadixInt FheContext::IsZero(const radix::RadixInt& a) {
    return radix::IsZero(a, *luts_);
}

radix::RadixInt FheContext::IsNonZero(const radix::RadixInt& a) {
    return radix::IsNonZero(a, *luts_);
}

// ============================================================================
// Type Casting
// ============================================================================

radix::RadixInt FheContext::Cast(const radix::RadixInt& a, FheType target_type) {
    auto params = GetRadixParams(target_type);
    return radix::Cast(a, params);
}

FheType FheContext::GetType(const radix::RadixInt& a) const {
    uint32_t bits = a.GetParams().total_bits();
    
    if (bits <= 1) return FheType::EBOOL;
    if (bits <= 4) return FheType::EUINT4;
    if (bits <= 8) return FheType::EUINT8;
    if (bits <= 16) return FheType::EUINT16;
    if (bits <= 32) return FheType::EUINT32;
    if (bits <= 64) return FheType::EUINT64;
    if (bits <= 128) return FheType::EUINT128;
    if (bits <= 160) return FheType::EUINT160;
    return FheType::EUINT256;
}

// ============================================================================
// Random Number Generation
// ============================================================================

radix::RadixInt FheContext::Random(FheType type) {
    // Use cryptographically secure random
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;
    
    uint64_t value = dis(gen);
    uint32_t bits = FheTypeBits(type);
    if (bits < 64) {
        value &= (1ULL << bits) - 1;
    }
    
    return Encrypt(KeyGen(), value, type);
}

radix::RadixInt FheContext::RandomRange(FheType type, uint64_t max) {
    auto random = Random(type);
    auto max_enc = TrivialEncrypt(max, type);
    return Rem(random, max_enc);
}

// ============================================================================
// Serialization
// ============================================================================

std::vector<uint8_t> FheContext::SerializeCiphertext(const radix::RadixInt& ct) {
    std::stringstream ss;
    
    // Serialize each limb
    for (size_t i = 0; i < ct.NumLimbs(); ++i) {
        Serial::Serialize(ct.GetLimb(i).GetCiphertext(), ss, SerType::BINARY);
    }
    
    std::string str = ss.str();
    return std::vector<uint8_t>(str.begin(), str.end());
}

radix::RadixInt FheContext::DeserializeCiphertext(
    const std::vector<uint8_t>& data,
    FheType type
) {
    auto params = GetRadixParams(type);
    radix::RadixInt result(*cc_, params);
    
    std::string str(data.begin(), data.end());
    std::stringstream ss(str);
    
    for (size_t i = 0; i < params.num_limbs; ++i) {
        LWECiphertext ct;
        Serial::Deserialize(ct, ss, SerType::BINARY);
        result.GetLimb(i).GetCiphertext() = ct;
    }
    
    return result;
}

std::vector<uint8_t> FheContext::SerializeSecretKey(const LWEPrivateKey& sk) {
    std::stringstream ss;
    Serial::Serialize(sk, ss, SerType::BINARY);
    std::string str = ss.str();
    return std::vector<uint8_t>(str.begin(), str.end());
}

LWEPrivateKey FheContext::DeserializeSecretKey(const std::vector<uint8_t>& data) {
    std::string str(data.begin(), data.end());
    std::stringstream ss(str);
    LWEPrivateKey sk;
    Serial::Deserialize(sk, ss, SerType::BINARY);
    return sk;
}

std::vector<uint8_t> FheContext::SerializePublicKey(const LWEPublicKey& pk) {
    std::stringstream ss;
    Serial::Serialize(pk, ss, SerType::BINARY);
    std::string str = ss.str();
    return std::vector<uint8_t>(str.begin(), str.end());
}

LWEPublicKey FheContext::DeserializePublicKey(const std::vector<uint8_t>& data) {
    std::string str(data.begin(), data.end());
    std::stringstream ss(str);
    LWEPublicKey pk;
    Serial::Deserialize(pk, ss, SerType::BINARY);
    return pk;
}

std::vector<uint8_t> FheContext::SerializeBootstrapKey() {
    std::stringstream ss;
    cc_->SerializeEvalKey(ss, SerType::BINARY);
    std::string str = ss.str();
    return std::vector<uint8_t>(str.begin(), str.end());
}

void FheContext::DeserializeBootstrapKey(const std::vector<uint8_t>& data) {
    std::string str(data.begin(), data.end());
    std::stringstream ss(str);
    cc_->DeserializeEvalKey(ss, SerType::BINARY);
}

// ============================================================================
// Verification
// ============================================================================

bool FheContext::Verify(const radix::RadixInt& ct) const {
    // Basic verification: check that ciphertexts are well-formed
    for (size_t i = 0; i < ct.NumLimbs(); ++i) {
        const auto& limb_ct = ct.GetLimb(i).GetCiphertext();
        if (limb_ct == nullptr) {
            return false;
        }
    }
    return true;
}

std::vector<uint8_t> FheContext::GetProof(const radix::RadixInt& ct) const {
    // Placeholder - would generate zero-knowledge proof of ciphertext validity
    return {};
}

bool FheContext::VerifyProof(
    const radix::RadixInt& ct,
    const std::vector<uint8_t>& proof
) const {
    // Placeholder - would verify zero-knowledge proof
    return Verify(ct);
}

// ============================================================================
// Batch Operations
// ============================================================================

void BatchAdd(
    FheContext& ctx,
    const std::vector<radix::RadixInt>& a,
    const std::vector<radix::RadixInt>& b,
    std::vector<radix::RadixInt>& results
) {
    radix::BatchAdd(a, b, results, ctx.GetLUTs());
}

void BatchMul(
    FheContext& ctx,
    const std::vector<radix::RadixInt>& a,
    const std::vector<radix::RadixInt>& b,
    std::vector<radix::RadixInt>& results
) {
    radix::BatchMul(a, b, results, ctx.GetLUTs());
}

void BatchLt(
    FheContext& ctx,
    const std::vector<radix::RadixInt>& a,
    const std::vector<radix::RadixInt>& b,
    std::vector<radix::RadixInt>& results
) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Input vectors must have same size");
    }
    
    results.resize(a.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < a.size(); ++i) {
        results[i] = ctx.Lt(a[i], b[i]);
    }
}

void BatchSelect(
    FheContext& ctx,
    const std::vector<radix::RadixInt>& cond,
    const std::vector<radix::RadixInt>& if_true,
    const std::vector<radix::RadixInt>& if_false,
    std::vector<radix::RadixInt>& results
) {
    if (cond.size() != if_true.size() || cond.size() != if_false.size()) {
        throw std::invalid_argument("Input vectors must have same size");
    }
    
    results.resize(cond.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < cond.size(); ++i) {
        results[i] = ctx.Select(cond[i], if_true[i], if_false[i]);
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

uint64_t EstimateGas(const std::string& op, FheType type) {
    // Gas estimates based on operation complexity
    // These should match the EVM precompile gas costs
    
    uint32_t bits = FheTypeBits(type);
    uint64_t base_cost = 10000;  // Base cost for any FHE op
    uint64_t bit_cost = 500;     // Cost per bit
    
    if (op == "add" || op == "sub") {
        return base_cost + bits * bit_cost;
    } else if (op == "mul") {
        return base_cost + bits * bits * bit_cost / 10;
    } else if (op == "div" || op == "rem") {
        return base_cost + bits * bits * bit_cost;
    } else if (op == "eq" || op == "ne" || op == "lt" || op == "le" || op == "gt" || op == "ge") {
        return base_cost + bits * bit_cost / 2;
    } else if (op == "and" || op == "or" || op == "xor" || op == "not") {
        return base_cost + bits * bit_cost / 4;
    } else if (op == "shl" || op == "shr" || op == "rotl" || op == "rotr") {
        return base_cost + bits * bit_cost;
    } else if (op == "select" || op == "cmux") {
        return base_cost + bits * bit_cost * 2;
    }
    
    return base_cost;
}

std::string Version() {
    return "Lux FHE 1.0.0 (OpenFHE 1.4.2)";
}

} // namespace fhevm
} // namespace lbcrypto
