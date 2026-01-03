// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// fhEVM Operations - Complete encrypted integer operations for EVM
//
// This module provides all operations required for fhEVM compatibility:
// - Encrypted integer types (euint4, euint8, ..., euint256, eaddress)
// - Arithmetic (add, sub, mul, div, rem)
// - Comparison (eq, ne, lt, le, gt, ge, min, max)
// - Bitwise (and, or, xor, not, shl, shr, rotl, rotr)
// - Control flow (select/cmux, if_then_else)
// - Type operations (cast, trivial_encrypt)
// - Key management (public key, serialization)

#ifndef FHEVM_FHEVM_H
#define FHEVM_FHEVM_H

#include "binfhecontext.h"
#include "radix/radix.h"
#include "radix/shortint.h"
#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <optional>

namespace lux::fhe {
namespace fhevm {

// ============================================================================
// Encrypted Integer Type Enumeration
// ============================================================================

enum class FheType : uint8_t {
    EBOOL = 0,      // 1-bit boolean
    EUINT4 = 1,     // 4-bit unsigned
    EUINT8 = 2,     // 8-bit unsigned
    EUINT16 = 3,    // 16-bit unsigned
    EUINT32 = 4,    // 32-bit unsigned
    EUINT64 = 5,    // 64-bit unsigned
    EUINT128 = 6,   // 128-bit unsigned
    EUINT160 = 7,   // 160-bit (Ethereum address)
    EUINT256 = 8,   // 256-bit (EVM word)
    EADDRESS = 7,   // Alias for EUINT160
};

// Get bit width for type
uint32_t FheTypeBits(FheType type);

// Get type name
std::string FheTypeName(FheType type);

// ============================================================================
// FheContext - Main context for fhEVM operations
// ============================================================================

class FheContext {
public:
    // Construct with security level
    explicit FheContext(BINFHE_PARAMSET params = STD128);
    ~FheContext();
    
    // Non-copyable, movable
    FheContext(const FheContext&) = delete;
    FheContext& operator=(const FheContext&) = delete;
    FheContext(FheContext&&) noexcept;
    FheContext& operator=(FheContext&&) noexcept;
    
    // ========================================================================
    // Key Generation
    // ========================================================================
    
    /**
     * @brief Generate secret key
     */
    LWEPrivateKey KeyGen();
    
    /**
     * @brief Generate public key from secret key
     */
    LWEPublicKey PublicKeyGen(const LWEPrivateKey& sk);
    
    /**
     * @brief Generate bootstrapping key (required for FHE operations)
     * This is expensive (~1-2 seconds) - should be done once and cached.
     */
    void BootstrapKeyGen(const LWEPrivateKey& sk);
    
    /**
     * @brief Generate key switching key
     */
    void SwitchKeyGen(const LWEPrivateKey& sk);
    
    /**
     * @brief Check if bootstrapping key is available
     */
    bool HasBootstrapKey() const;
    
    // ========================================================================
    // Encryption / Decryption
    // ========================================================================
    
    /**
     * @brief Encrypt with secret key
     * @param sk Secret key
     * @param value Plaintext value (truncated to type width)
     * @param type Encrypted type
     */
    radix::RadixInt Encrypt(
        const LWEPrivateKey& sk,
        uint64_t value,
        FheType type
    );
    
    /**
     * @brief Encrypt with secret key from bytes
     * @param sk Secret key
     * @param bytes Big-endian byte representation
     * @param type Encrypted type
     */
    radix::RadixInt EncryptBytes(
        const LWEPrivateKey& sk,
        const std::vector<uint8_t>& bytes,
        FheType type
    );
    
    /**
     * @brief Encrypt with public key (compact ciphertext)
     * @param pk Public key
     * @param value Plaintext value
     * @param type Encrypted type
     */
    radix::RadixInt EncryptPublic(
        const LWEPublicKey& pk,
        uint64_t value,
        FheType type
    );
    
    /**
     * @brief Trivial encryption (plaintext as ciphertext, no security)
     * Used for public constants in computation.
     */
    radix::RadixInt TrivialEncrypt(uint64_t value, FheType type);
    
    /**
     * @brief Trivial encryption from bytes
     */
    radix::RadixInt TrivialEncryptBytes(
        const std::vector<uint8_t>& bytes,
        FheType type
    );
    
    /**
     * @brief Decrypt to uint64
     * @note Only works for types <= 64 bits
     */
    uint64_t Decrypt(const LWEPrivateKey& sk, const radix::RadixInt& ct);
    
    /**
     * @brief Decrypt to bytes (big-endian)
     */
    std::vector<uint8_t> DecryptBytes(
        const LWEPrivateKey& sk,
        const radix::RadixInt& ct
    );
    
    // ========================================================================
    // Arithmetic Operations
    // ========================================================================
    
    /**
     * @brief Add two encrypted values
     * @return a + b (mod 2^width)
     */
    radix::RadixInt Add(const radix::RadixInt& a, const radix::RadixInt& b);
    
    /**
     * @brief Add encrypted value and plaintext
     */
    radix::RadixInt AddScalar(const radix::RadixInt& a, uint64_t b);
    
    /**
     * @brief Subtract two encrypted values
     * @return a - b (mod 2^width)
     */
    radix::RadixInt Sub(const radix::RadixInt& a, const radix::RadixInt& b);
    
    /**
     * @brief Subtract plaintext from encrypted
     */
    radix::RadixInt SubScalar(const radix::RadixInt& a, uint64_t b);
    
    /**
     * @brief Negate encrypted value
     * @return -a (two's complement)
     */
    radix::RadixInt Neg(const radix::RadixInt& a);
    
    /**
     * @brief Multiply two encrypted values
     * @return (a * b) mod 2^width (truncated)
     */
    radix::RadixInt Mul(const radix::RadixInt& a, const radix::RadixInt& b);
    
    /**
     * @brief Multiply encrypted by plaintext
     */
    radix::RadixInt MulScalar(const radix::RadixInt& a, uint64_t b);
    
    /**
     * @brief Divide two encrypted values
     * @return a / b (integer division)
     */
    radix::RadixInt Div(const radix::RadixInt& a, const radix::RadixInt& b);
    
    /**
     * @brief Divide encrypted by plaintext
     */
    radix::RadixInt DivScalar(const radix::RadixInt& a, uint64_t b);
    
    /**
     * @brief Remainder of division
     * @return a % b
     */
    radix::RadixInt Rem(const radix::RadixInt& a, const radix::RadixInt& b);
    
    /**
     * @brief Remainder with plaintext divisor
     */
    radix::RadixInt RemScalar(const radix::RadixInt& a, uint64_t b);
    
    // ========================================================================
    // Comparison Operations
    // ========================================================================
    
    /**
     * @brief Equal comparison
     * @return encrypted 1 if a == b, 0 otherwise
     */
    radix::RadixInt Eq(const radix::RadixInt& a, const radix::RadixInt& b);
    
    /**
     * @brief Not equal comparison
     */
    radix::RadixInt Ne(const radix::RadixInt& a, const radix::RadixInt& b);
    
    /**
     * @brief Less than comparison
     * @return encrypted 1 if a < b, 0 otherwise
     */
    radix::RadixInt Lt(const radix::RadixInt& a, const radix::RadixInt& b);
    
    /**
     * @brief Less than or equal comparison
     */
    radix::RadixInt Le(const radix::RadixInt& a, const radix::RadixInt& b);
    
    /**
     * @brief Greater than comparison
     */
    radix::RadixInt Gt(const radix::RadixInt& a, const radix::RadixInt& b);
    
    /**
     * @brief Greater than or equal comparison
     */
    radix::RadixInt Ge(const radix::RadixInt& a, const radix::RadixInt& b);
    
    /**
     * @brief Minimum of two values
     */
    radix::RadixInt Min(const radix::RadixInt& a, const radix::RadixInt& b);
    
    /**
     * @brief Maximum of two values
     */
    radix::RadixInt Max(const radix::RadixInt& a, const radix::RadixInt& b);
    
    // Scalar comparison variants
    radix::RadixInt EqScalar(const radix::RadixInt& a, uint64_t b);
    radix::RadixInt NeScalar(const radix::RadixInt& a, uint64_t b);
    radix::RadixInt LtScalar(const radix::RadixInt& a, uint64_t b);
    radix::RadixInt LeScalar(const radix::RadixInt& a, uint64_t b);
    radix::RadixInt GtScalar(const radix::RadixInt& a, uint64_t b);
    radix::RadixInt GeScalar(const radix::RadixInt& a, uint64_t b);
    
    // ========================================================================
    // Bitwise Operations
    // ========================================================================
    
    /**
     * @brief Bitwise AND
     */
    radix::RadixInt And(const radix::RadixInt& a, const radix::RadixInt& b);
    
    /**
     * @brief Bitwise OR
     */
    radix::RadixInt Or(const radix::RadixInt& a, const radix::RadixInt& b);
    
    /**
     * @brief Bitwise XOR
     */
    radix::RadixInt Xor(const radix::RadixInt& a, const radix::RadixInt& b);
    
    /**
     * @brief Bitwise NOT
     */
    radix::RadixInt Not(const radix::RadixInt& a);
    
    // Scalar bitwise variants
    radix::RadixInt AndScalar(const radix::RadixInt& a, uint64_t b);
    radix::RadixInt OrScalar(const radix::RadixInt& a, uint64_t b);
    radix::RadixInt XorScalar(const radix::RadixInt& a, uint64_t b);
    
    // ========================================================================
    // Shift and Rotate Operations
    // ========================================================================
    
    /**
     * @brief Left shift by encrypted amount
     */
    radix::RadixInt Shl(const radix::RadixInt& a, const radix::RadixInt& bits);
    
    /**
     * @brief Left shift by constant amount
     */
    radix::RadixInt ShlScalar(const radix::RadixInt& a, uint32_t bits);
    
    /**
     * @brief Right shift by encrypted amount (logical, zero-fill)
     */
    radix::RadixInt Shr(const radix::RadixInt& a, const radix::RadixInt& bits);
    
    /**
     * @brief Right shift by constant amount
     */
    radix::RadixInt ShrScalar(const radix::RadixInt& a, uint32_t bits);
    
    /**
     * @brief Rotate left by encrypted amount
     */
    radix::RadixInt Rotl(const radix::RadixInt& a, const radix::RadixInt& bits);
    
    /**
     * @brief Rotate left by constant amount
     */
    radix::RadixInt RotlScalar(const radix::RadixInt& a, uint32_t bits);
    
    /**
     * @brief Rotate right by encrypted amount
     */
    radix::RadixInt Rotr(const radix::RadixInt& a, const radix::RadixInt& bits);
    
    /**
     * @brief Rotate right by constant amount
     */
    radix::RadixInt RotrScalar(const radix::RadixInt& a, uint32_t bits);
    
    // ========================================================================
    // Control Flow Operations
    // ========================================================================
    
    /**
     * @brief Conditional select (CMUX)
     * @param cond Encrypted boolean condition
     * @param if_true Value if cond is true (non-zero)
     * @param if_false Value if cond is false (zero)
     * @return if_true if cond != 0, else if_false
     */
    radix::RadixInt Select(
        const radix::RadixInt& cond,
        const radix::RadixInt& if_true,
        const radix::RadixInt& if_false
    );
    
    /**
     * @brief Conditional select with encrypted boolean
     */
    radix::RadixInt IfThenElse(
        const radix::RadixInt& cond,
        const radix::RadixInt& then_val,
        const radix::RadixInt& else_val
    );
    
    /**
     * @brief Check if value is zero
     * @return encrypted 1 if a == 0, 0 otherwise
     */
    radix::RadixInt IsZero(const radix::RadixInt& a);
    
    /**
     * @brief Check if value is non-zero
     */
    radix::RadixInt IsNonZero(const radix::RadixInt& a);
    
    // ========================================================================
    // Type Casting
    // ========================================================================
    
    /**
     * @brief Cast to different encrypted type
     * Truncates if target is smaller, zero-extends if larger.
     */
    radix::RadixInt Cast(const radix::RadixInt& a, FheType target_type);
    
    /**
     * @brief Get type of encrypted value
     */
    FheType GetType(const radix::RadixInt& a) const;
    
    // ========================================================================
    // Random Number Generation
    // ========================================================================
    
    /**
     * @brief Generate random encrypted value
     * Uses secure randomness internally.
     */
    radix::RadixInt Random(FheType type);
    
    /**
     * @brief Generate random encrypted value in range [0, max)
     */
    radix::RadixInt RandomRange(FheType type, uint64_t max);
    
    // ========================================================================
    // Serialization
    // ========================================================================
    
    /**
     * @brief Serialize ciphertext to bytes
     */
    std::vector<uint8_t> SerializeCiphertext(const radix::RadixInt& ct);
    
    /**
     * @brief Deserialize ciphertext from bytes
     */
    radix::RadixInt DeserializeCiphertext(
        const std::vector<uint8_t>& data,
        FheType type
    );
    
    /**
     * @brief Serialize secret key
     */
    std::vector<uint8_t> SerializeSecretKey(const LWEPrivateKey& sk);
    
    /**
     * @brief Deserialize secret key
     */
    LWEPrivateKey DeserializeSecretKey(const std::vector<uint8_t>& data);
    
    /**
     * @brief Serialize public key
     */
    std::vector<uint8_t> SerializePublicKey(const LWEPublicKey& pk);
    
    /**
     * @brief Deserialize public key
     */
    LWEPublicKey DeserializePublicKey(const std::vector<uint8_t>& data);
    
    /**
     * @brief Serialize bootstrapping key (large, ~100MB)
     */
    std::vector<uint8_t> SerializeBootstrapKey();
    
    /**
     * @brief Deserialize bootstrapping key
     */
    void DeserializeBootstrapKey(const std::vector<uint8_t>& data);
    
    // ========================================================================
    // Verification (for on-chain verification)
    // ========================================================================
    
    /**
     * @brief Verify ciphertext is well-formed
     */
    bool Verify(const radix::RadixInt& ct) const;
    
    /**
     * @brief Get compact proof of ciphertext validity
     */
    std::vector<uint8_t> GetProof(const radix::RadixInt& ct) const;
    
    /**
     * @brief Verify proof
     */
    bool VerifyProof(
        const radix::RadixInt& ct,
        const std::vector<uint8_t>& proof
    ) const;
    
    // ========================================================================
    // Accessors
    // ========================================================================
    
    BinFHEContext& GetContext() { return *cc_; }
    const BinFHEContext& GetContext() const { return *cc_; }
    
    const radix::ShortIntLUTs& GetLUTs() const { return *luts_; }
    
private:
    std::unique_ptr<BinFHEContext> cc_;
    std::unique_ptr<radix::ShortIntLUTs> luts_;
    
    // Get radix params for type
    radix::RadixParams GetRadixParams(FheType type) const;
};

// ============================================================================
// Batch Operations (for GPU throughput)
// ============================================================================

/**
 * @brief Batch add
 */
void BatchAdd(
    FheContext& ctx,
    const std::vector<radix::RadixInt>& a,
    const std::vector<radix::RadixInt>& b,
    std::vector<radix::RadixInt>& results
);

/**
 * @brief Batch multiply
 */
void BatchMul(
    FheContext& ctx,
    const std::vector<radix::RadixInt>& a,
    const std::vector<radix::RadixInt>& b,
    std::vector<radix::RadixInt>& results
);

/**
 * @brief Batch comparison
 */
void BatchLt(
    FheContext& ctx,
    const std::vector<radix::RadixInt>& a,
    const std::vector<radix::RadixInt>& b,
    std::vector<radix::RadixInt>& results
);

/**
 * @brief Batch select
 */
void BatchSelect(
    FheContext& ctx,
    const std::vector<radix::RadixInt>& cond,
    const std::vector<radix::RadixInt>& if_true,
    const std::vector<radix::RadixInt>& if_false,
    std::vector<radix::RadixInt>& results
);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Estimate gas cost for operation
 */
uint64_t EstimateGas(const std::string& op, FheType type);

/**
 * @brief Get library version
 */
std::string Version();

} // namespace fhevm
} // namespace lux::fhe

#endif // FHEVM_FHEVM_H
