// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// Packed Key/Ciphertext Formats for GPU Transfer
// 
// Defines contiguous memory layouts optimized for:
// - Bulk DMA transfer to GPU
// - Coalesced memory access patterns in kernels
// - Minimal serialization overhead

#ifndef BACKEND_PACKED_H
#define BACKEND_PACKED_H

#include "binfhecontext.h"
#include "lwe-ciphertext.h"
#include "rgsw-acc.h"
#include <cstdint>
#include <vector>
#include <memory>

namespace lux::fhe {
namespace backend {

// ============================================================================
// Packed Format Headers
// ============================================================================

// Magic bytes for format identification
constexpr uint32_t PACKED_MAGIC = 0x4C555846; // "LUXF"
constexpr uint16_t PACKED_VERSION = 1;

// Format type identifiers
enum class PackedType : uint16_t {
    LWE_CIPHERTEXT = 1,
    LWE_BATCH = 2,
    RLWE_CIPHERTEXT = 3,
    RGSW_CIPHERTEXT = 4,
    BOOTSTRAPPING_KEY = 5,
    SWITCHING_KEY = 6,
    PARAMETERS = 7
};

// Common header for all packed formats
struct PackedHeader {
    uint32_t magic;           // PACKED_MAGIC
    uint16_t version;         // PACKED_VERSION
    PackedType type;          // What's packed
    uint64_t total_size;      // Total size in bytes including header
    uint64_t element_count;   // Number of elements (for batches)
    uint32_t flags;           // Format-specific flags
    uint32_t reserved;        // Padding/future use
};
static_assert(sizeof(PackedHeader) == 32, "PackedHeader must be 32 bytes");

// ============================================================================
// LWE Ciphertext Packed Format
// ============================================================================

// Flags for LWE packing
enum LWEPackFlags : uint32_t {
    LWE_PACK_DEFAULT = 0,
    LWE_PACK_INTERLEAVED = 1 << 0,  // Interleave coefficients for coalescing
    LWE_PACK_COMPRESSED = 1 << 1,   // Apply compression
};

/**
 * @brief Packed format for a single LWE ciphertext
 * 
 * Layout: [header][params][a_vector][b_scalar]
 * 
 * The 'a' vector has n coefficients, 'b' is the scalar part.
 * For batches, multiple ciphertexts are concatenated.
 */
struct PackedLWE {
    PackedHeader header;
    uint32_t n;               // LWE dimension
    uint32_t log_q;           // Log of modulus
    uint64_t q;               // Modulus
    // Followed by:
    // - NativeInteger a[n]
    // - NativeInteger b
};

/**
 * @brief Packed batch of LWE ciphertexts
 * 
 * For GPU efficiency, batch layout can be:
 * - Sequential: [ct1][ct2][ct3]... (default, good for independent ops)
 * - Interleaved: [a1[0],a2[0],...][a1[1],a2[1],...]... (good for SIMD)
 */
struct PackedLWEBatch {
    PackedHeader header;
    uint32_t n;               // LWE dimension
    uint32_t log_q;           // Log of modulus
    uint64_t q;               // Modulus
    uint64_t count;           // Number of ciphertexts
    uint32_t stride;          // Bytes between elements (for interleaved)
    uint32_t reserved;
    // Followed by packed ciphertexts
};

// ============================================================================
// RLWE/RGSW Packed Formats
// ============================================================================

/**
 * @brief Packed RLWE ciphertext (polynomial pair)
 * 
 * RLWE = (a, b) where a, b ∈ R_Q
 * Layout: [header][params][a_coeffs][b_coeffs]
 */
struct PackedRLWE {
    PackedHeader header;
    uint32_t N;               // Ring dimension (power of 2)
    uint32_t num_limbs;       // RNS decomposition limbs
    uint64_t Q;               // Ciphertext modulus
    // Followed by:
    // - Coefficients for polynomial a (N × num_limbs)
    // - Coefficients for polynomial b (N × num_limbs)
};

/**
 * @brief Packed RGSW ciphertext (gadget matrix)
 * 
 * RGSW is a matrix of RLWE ciphertexts.
 * Layout: [header][params][rlwe_matrix]
 */
struct PackedRGSW {
    PackedHeader header;
    uint32_t N;               // Ring dimension
    uint32_t num_limbs;       // RNS limbs
    uint32_t rows;            // Matrix rows (2 × decomposition levels)
    uint32_t cols;            // Matrix columns (typically 2)
    uint64_t Q;               // Ciphertext modulus
    // Followed by rows × cols RLWE ciphertexts
};

// ============================================================================
// Bootstrapping Key Packed Format
// ============================================================================

/**
 * @brief Packed bootstrapping key for GPU
 * 
 * The bootstrapping key is a collection of RGSW ciphertexts, one per
 * LWE secret key coefficient. This is the largest structure and the
 * primary target for GPU memory optimization.
 * 
 * Size estimate: For n=570, N=1024, decomp_levels=3
 *   = 570 × 2 × 3 × 2 × 1024 × 8 bytes ≈ 140 MB
 * 
 * We use a packed layout that enables efficient blind rotation:
 * - Coefficients grouped for polynomial multiplication
 * - RNS limbs arranged for parallel NTT
 */
struct PackedBootstrappingKey {
    PackedHeader header;
    
    // LWE parameters
    uint32_t lwe_n;           // LWE dimension
    uint32_t lwe_log_q;       // LWE modulus bits
    
    // RLWE parameters
    uint32_t rlwe_N;          // Ring dimension
    uint32_t rlwe_num_limbs;  // RNS limbs
    
    // Decomposition parameters
    uint32_t decomp_levels;   // Gadget decomposition levels
    uint32_t decomp_base_log; // Base-2 log of decomposition base
    
    // Key layout
    uint64_t key_size;        // Total size of packed key data
    uint32_t key_layout;      // Layout flags
    uint32_t reserved;
    
    // Followed by:
    // - Packed RGSW ciphertexts for each LWE key coefficient
    // - Layout optimized for blind rotation access pattern
};

// Key layout options
enum KeyLayoutFlags : uint32_t {
    KEY_LAYOUT_STANDARD = 0,      // Standard RGSW matrix layout
    KEY_LAYOUT_ROTATED = 1 << 0,  // Pre-rotated for automorphism
    KEY_LAYOUT_NTT = 1 << 1,      // Stored in NTT domain
    KEY_LAYOUT_SPLIT = 1 << 2,    // Split for multi-GPU
};

// ============================================================================
// Key Switching Key Packed Format
// ============================================================================

/**
 * @brief Packed key switching key
 * 
 * For switching from one LWE key to another.
 */
struct PackedSwitchingKey {
    PackedHeader header;
    uint32_t input_n;         // Input LWE dimension
    uint32_t output_n;        // Output LWE dimension
    uint32_t decomp_levels;   // Key switching levels
    uint32_t decomp_base_log; // Decomposition base
    uint64_t Q;               // Modulus
    uint32_t reserved[2];
    // Followed by key switching matrix
};

// ============================================================================
// Packing/Unpacking Functions
// ============================================================================

/**
 * @brief Pack a single LWE ciphertext
 */
std::vector<uint8_t> PackLWE(
    const LWECiphertext& ct,
    uint32_t flags = LWE_PACK_DEFAULT
);

/**
 * @brief Unpack a single LWE ciphertext
 */
LWECiphertext UnpackLWE(const uint8_t* data, size_t size);

/**
 * @brief Pack a batch of LWE ciphertexts
 */
std::vector<uint8_t> PackLWEBatch(
    const std::vector<LWECiphertext>& cts,
    uint32_t flags = LWE_PACK_DEFAULT
);

/**
 * @brief Unpack a batch of LWE ciphertexts
 */
std::vector<LWECiphertext> UnpackLWEBatch(const uint8_t* data, size_t size);

/**
 * @brief Pack bootstrapping key
 * 
 * This is the expensive operation but only done once per context.
 */
std::vector<uint8_t> PackBootstrappingKey(
    const RingGSWACCKey& ek,
    const std::shared_ptr<RingGSWCryptoParams>& params,
    uint32_t flags = KEY_LAYOUT_STANDARD
);

/**
 * @brief Unpack bootstrapping key
 */
RingGSWACCKey UnpackBootstrappingKey(
    const uint8_t* data,
    size_t size,
    const std::shared_ptr<RingGSWCryptoParams>& params
);

/**
 * @brief Pack key switching key
 */
std::vector<uint8_t> PackSwitchingKey(
    const LWESwitchingKey& ks,
    const std::shared_ptr<LWECryptoParams>& params
);

/**
 * @brief Unpack key switching key
 */
LWESwitchingKey UnpackSwitchingKey(
    const uint8_t* data,
    size_t size,
    const std::shared_ptr<LWECryptoParams>& params
);

// ============================================================================
// Size Estimation
// ============================================================================

/**
 * @brief Estimate packed size for LWE batch
 */
size_t EstimatePackedLWEBatchSize(
    uint32_t count,
    uint32_t n,
    uint32_t log_q
);

/**
 * @brief Estimate packed size for bootstrapping key
 */
size_t EstimatePackedBootstrappingKeySize(
    uint32_t lwe_n,
    uint32_t rlwe_N,
    uint32_t rlwe_limbs,
    uint32_t decomp_levels
);

/**
 * @brief Check if data has valid packed format
 */
bool ValidatePackedHeader(const uint8_t* data, size_t size);

/**
 * @brief Get type from packed data
 */
PackedType GetPackedType(const uint8_t* data, size_t size);

} // namespace backend
} // namespace lux::fhe

#endif // BACKEND_PACKED_H
