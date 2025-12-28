// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// Packed format implementation for GPU transfer

#include "backend/packed.h"
#include <cstring>
#include <stdexcept>

namespace lbcrypto {
namespace backend {

// ============================================================================
// Validation
// ============================================================================

bool ValidatePackedHeader(const uint8_t* data, size_t size) {
    if (size < sizeof(PackedHeader)) {
        return false;
    }
    
    const auto* header = reinterpret_cast<const PackedHeader*>(data);
    if (header->magic != PACKED_MAGIC) {
        return false;
    }
    if (header->version > PACKED_VERSION) {
        return false;
    }
    if (header->total_size > size) {
        return false;
    }
    
    return true;
}

PackedType GetPackedType(const uint8_t* data, size_t size) {
    if (!ValidatePackedHeader(data, size)) {
        throw std::invalid_argument("Invalid packed data header");
    }
    
    const auto* header = reinterpret_cast<const PackedHeader*>(data);
    return header->type;
}

// ============================================================================
// Size Estimation
// ============================================================================

size_t EstimatePackedLWEBatchSize(uint32_t count, uint32_t n, uint32_t log_q) {
    // Header + params + count * (n+1) coefficients
    size_t coeff_size = (log_q + 7) / 8;  // Bytes per coefficient
    return sizeof(PackedLWEBatch) + count * (n + 1) * coeff_size;
}

size_t EstimatePackedBootstrappingKeySize(
    uint32_t lwe_n,
    uint32_t rlwe_N,
    uint32_t rlwe_limbs,
    uint32_t decomp_levels
) {
    // Each RGSW is 2 * decomp_levels RLWEs
    // Each RLWE is 2 polynomials of degree N
    // We have lwe_n RGSWs (one per secret key bit)
    size_t rlwe_size = 2 * rlwe_N * rlwe_limbs * sizeof(uint64_t);
    size_t rgsw_size = 2 * decomp_levels * rlwe_size;
    return sizeof(PackedBootstrappingKey) + lwe_n * rgsw_size;
}

// ============================================================================
// LWE Packing
// ============================================================================

std::vector<uint8_t> PackLWE(const LWECiphertext& ct, uint32_t flags) {
    const auto& a = ct->GetA();
    const auto& b = ct->GetB();
    uint32_t n = a.GetLength();
    
    // Calculate size
    size_t coeff_size = sizeof(NativeInteger);
    size_t total_size = sizeof(PackedLWE) + (n + 1) * coeff_size;
    
    std::vector<uint8_t> result(total_size);
    auto* packed = reinterpret_cast<PackedLWE*>(result.data());
    
    // Fill header
    packed->header.magic = PACKED_MAGIC;
    packed->header.version = PACKED_VERSION;
    packed->header.type = PackedType::LWE_CIPHERTEXT;
    packed->header.total_size = total_size;
    packed->header.element_count = 1;
    packed->header.flags = flags;
    packed->header.reserved = 0;
    
    // Fill params
    packed->n = n;
    packed->log_q = 64;  // TODO: Get actual modulus info
    packed->q = 0;  // Would need actual modulus
    
    // Copy coefficients
    uint8_t* coeff_ptr = result.data() + sizeof(PackedLWE);
    std::memcpy(coeff_ptr, a.GetData(), n * coeff_size);
    std::memcpy(coeff_ptr + n * coeff_size, &b, coeff_size);
    
    return result;
}

LWECiphertext UnpackLWE(const uint8_t* data, size_t size) {
    if (!ValidatePackedHeader(data, size)) {
        throw std::invalid_argument("Invalid packed LWE data");
    }
    
    const auto* packed = reinterpret_cast<const PackedLWE*>(data);
    if (packed->header.type != PackedType::LWE_CIPHERTEXT) {
        throw std::invalid_argument("Data is not a packed LWE ciphertext");
    }
    
    uint32_t n = packed->n;
    const uint8_t* coeff_ptr = data + sizeof(PackedLWE);
    
    // Create LWE ciphertext
    NativeVector a(n);
    std::memcpy(a.GetData(), coeff_ptr, n * sizeof(NativeInteger));
    
    NativeInteger b;
    std::memcpy(&b, coeff_ptr + n * sizeof(NativeInteger), sizeof(NativeInteger));
    
    return std::make_shared<LWECiphertextImpl>(std::move(a), std::move(b));
}

// ============================================================================
// LWE Batch Packing
// ============================================================================

std::vector<uint8_t> PackLWEBatch(
    const std::vector<LWECiphertext>& cts,
    uint32_t flags
) {
    if (cts.empty()) {
        return {};
    }
    
    uint32_t n = cts[0]->GetA().GetLength();
    size_t coeff_size = sizeof(NativeInteger);
    size_t ct_size = (n + 1) * coeff_size;
    size_t total_size = sizeof(PackedLWEBatch) + cts.size() * ct_size;
    
    std::vector<uint8_t> result(total_size);
    auto* packed = reinterpret_cast<PackedLWEBatch*>(result.data());
    
    // Fill header
    packed->header.magic = PACKED_MAGIC;
    packed->header.version = PACKED_VERSION;
    packed->header.type = PackedType::LWE_BATCH;
    packed->header.total_size = total_size;
    packed->header.element_count = cts.size();
    packed->header.flags = flags;
    packed->header.reserved = 0;
    
    // Fill params
    packed->n = n;
    packed->log_q = 64;
    packed->q = 0;
    packed->count = cts.size();
    packed->stride = ct_size;
    packed->reserved = 0;
    
    // Pack ciphertexts
    uint8_t* ptr = result.data() + sizeof(PackedLWEBatch);
    
    if (flags & LWE_PACK_INTERLEAVED) {
        // Interleaved layout: group by coefficient index
        // [a0[0], a1[0], a2[0], ...][a0[1], a1[1], ...][b0, b1, ...]
        for (uint32_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < cts.size(); ++i) {
                std::memcpy(ptr, &cts[i]->GetA()[j], coeff_size);
                ptr += coeff_size;
            }
        }
        for (size_t i = 0; i < cts.size(); ++i) {
            const auto& b = cts[i]->GetB();
            std::memcpy(ptr, &b, coeff_size);
            ptr += coeff_size;
        }
    } else {
        // Sequential layout: [ct0][ct1][ct2]...
        for (const auto& ct : cts) {
            const auto& a = ct->GetA();
            const auto& b = ct->GetB();
            std::memcpy(ptr, a.GetData(), n * coeff_size);
            ptr += n * coeff_size;
            std::memcpy(ptr, &b, coeff_size);
            ptr += coeff_size;
        }
    }
    
    return result;
}

std::vector<LWECiphertext> UnpackLWEBatch(const uint8_t* data, size_t size) {
    if (!ValidatePackedHeader(data, size)) {
        throw std::invalid_argument("Invalid packed LWE batch data");
    }
    
    const auto* packed = reinterpret_cast<const PackedLWEBatch*>(data);
    if (packed->header.type != PackedType::LWE_BATCH) {
        throw std::invalid_argument("Data is not a packed LWE batch");
    }
    
    uint32_t n = packed->n;
    uint64_t count = packed->count;
    size_t coeff_size = sizeof(NativeInteger);
    
    std::vector<LWECiphertext> result;
    result.reserve(count);
    
    const uint8_t* ptr = data + sizeof(PackedLWEBatch);
    
    if (packed->header.flags & LWE_PACK_INTERLEAVED) {
        // Deinterleave
        std::vector<NativeVector> a_vecs(count, NativeVector(n));
        std::vector<NativeInteger> b_vals(count);
        
        for (uint32_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < count; ++i) {
                std::memcpy(&a_vecs[i][j], ptr, coeff_size);
                ptr += coeff_size;
            }
        }
        for (size_t i = 0; i < count; ++i) {
            std::memcpy(&b_vals[i], ptr, coeff_size);
            ptr += coeff_size;
        }
        
        for (size_t i = 0; i < count; ++i) {
            result.push_back(std::make_shared<LWECiphertextImpl>(
                std::move(a_vecs[i]), std::move(b_vals[i])
            ));
        }
    } else {
        // Sequential
        for (size_t i = 0; i < count; ++i) {
            NativeVector a(n);
            std::memcpy(a.GetData(), ptr, n * coeff_size);
            ptr += n * coeff_size;
            
            NativeInteger b;
            std::memcpy(&b, ptr, coeff_size);
            ptr += coeff_size;
            
            result.push_back(std::make_shared<LWECiphertextImpl>(
                std::move(a), std::move(b)
            ));
        }
    }
    
    return result;
}

// ============================================================================
// Bootstrapping Key Packing
// ============================================================================

std::vector<uint8_t> PackBootstrappingKey(
    const RingGSWACCKey& ek,
    const std::shared_ptr<RingGSWCryptoParams>& params,
    uint32_t flags
) {
    // This is a complex operation that serializes the entire BT key
    // For now, return empty - real implementation would:
    // 1. Extract all RGSW ciphertexts from ek
    // 2. Convert to NTT form if KEY_LAYOUT_NTT
    // 3. Pack into contiguous buffer
    
    // TODO: Implement full BT key serialization
    std::vector<uint8_t> result;
    return result;
}

RingGSWACCKey UnpackBootstrappingKey(
    const uint8_t* data,
    size_t size,
    const std::shared_ptr<RingGSWCryptoParams>& params
) {
    // TODO: Implement full BT key deserialization
    return nullptr;
}

// ============================================================================
// Key Switching Key Packing
// ============================================================================

std::vector<uint8_t> PackSwitchingKey(
    const LWESwitchingKey& ks,
    const std::shared_ptr<LWECryptoParams>& params
) {
    // TODO: Implement
    return {};
}

LWESwitchingKey UnpackSwitchingKey(
    const uint8_t* data,
    size_t size,
    const std::shared_ptr<LWECryptoParams>& params
) {
    // TODO: Implement
    return nullptr;
}

} // namespace backend
} // namespace lbcrypto
