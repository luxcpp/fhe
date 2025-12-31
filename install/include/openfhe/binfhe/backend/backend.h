// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// Backend Abstraction - Seam between BinFHE operations and compute kernels
// 
// The existing OpenFHE GINX implementation is the CPU baseline. This interface
// allows plugging in GPU backends (MLX for Apple Silicon).
//
// NVIDIA CUDA support: Contact licensing@lux.network for enterprise versions.

#ifndef BACKEND_BACKEND_H
#define BACKEND_BACKEND_H

#include "binfhecontext.h"
#include "lwe-ciphertext.h"
#include "rgsw-acc.h"
#include "rgsw-evalkey.h"
#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <functional>

namespace lbcrypto {
namespace backend {

// ============================================================================
// Backend Type Enumeration
// ============================================================================

enum class BackendType {
    CPU,        // Default: OpenFHE GINX (existing code, correctness baseline)
    MLX,        // Apple Silicon GPU via MLX
    CUDA,       // Enterprise only - contact licensing@lux.network
    AUTO        // Auto-select best available (CPU or MLX)
};

std::string BackendTypeName(BackendType type);
BackendType BackendTypeFromName(const std::string& name);

// ============================================================================
// Device Memory Handles
// ============================================================================

// Opaque handle for device-side data (GPU memory)
// On CPU backend, this just wraps the host pointer
struct DeviceBuffer {
    void* ptr = nullptr;
    size_t size = 0;
    BackendType device = BackendType::CPU;
    
    bool IsValid() const { return ptr != nullptr && size > 0; }
    bool IsDevice() const { return device != BackendType::CPU; }
};

// ============================================================================
// Backend Interface - The Seam
// ============================================================================

/**
 * @brief Abstract interface for FHE compute backends
 * 
 * This defines the hot operations that need GPU acceleration:
 * - Blind rotation (accumulator update)
 * - External product (RGSW × RLWE)
 * - Gadget decomposition
 * - Key switching
 * - Modulus switching
 * 
 * The CPU backend simply calls existing OpenFHE GINX code.
 * GPU backends implement these with CUDA/MLX kernels.
 */
class Backend {
public:
    virtual ~Backend() = default;
    
    // ========================================================================
    // Backend Info
    // ========================================================================
    
    virtual BackendType Type() const = 0;
    virtual std::string Name() const = 0;
    virtual bool IsAvailable() const = 0;
    
    // Device capabilities
    virtual size_t MaxBatchSize() const = 0;
    virtual size_t DeviceMemory() const = 0;  // In bytes, 0 for CPU
    
    // ========================================================================
    // Memory Management
    // ========================================================================
    
    // Allocate device memory
    virtual DeviceBuffer Allocate(size_t bytes) = 0;
    
    // Free device memory
    virtual void Free(DeviceBuffer& buffer) = 0;
    
    // Copy host → device
    virtual void CopyToDevice(
        const void* host_ptr,
        DeviceBuffer& device_buffer,
        size_t bytes
    ) = 0;
    
    // Copy device → host
    virtual void CopyToHost(
        const DeviceBuffer& device_buffer,
        void* host_ptr,
        size_t bytes
    ) = 0;
    
    // Synchronize (wait for async operations)
    virtual void Synchronize() = 0;
    
    // ========================================================================
    // Core FHE Operations (Single)
    // ========================================================================
    
    /**
     * @brief Blind rotation - the core bootstrapping operation
     * 
     * Updates the accumulator based on the LWE ciphertext and bootstrapping key.
     * This is where GINX lives.
     * 
     * @param params Crypto parameters
     * @param ct Input LWE ciphertext
     * @param ek Evaluation key (bootstrapping key)
     * @param acc Accumulator (in/out)
     */
    virtual void BlindRotate(
        const std::shared_ptr<RingGSWCryptoParams>& params,
        const LWECiphertext& ct,
        const RingGSWACCKey& ek,
        RLWECiphertext& acc
    ) = 0;
    
    /**
     * @brief External product: RGSW × RLWE → RLWE
     */
    virtual void ExternalProduct(
        const std::shared_ptr<RingGSWCryptoParams>& params,
        const RingGSWEvalKey& rgsw,
        const RLWECiphertext& rlwe,
        RLWECiphertext& result
    ) = 0;
    
    /**
     * @brief Key switching: LWE_{s1} → LWE_{s2}
     */
    virtual void KeySwitch(
        const std::shared_ptr<LWECryptoParams>& params,
        const LWECiphertext& ct,
        const LWESwitchingKey& ks,
        LWECiphertext& result
    ) = 0;
    
    /**
     * @brief Modulus switching: LWE_{q1} → LWE_{q2}
     */
    virtual void ModSwitch(
        const std::shared_ptr<LWECryptoParams>& params,
        const LWECiphertext& ct,
        LWECiphertext& result
    ) = 0;
    
    // ========================================================================
    // Batch Operations (GPU hot path)
    // ========================================================================
    
    /**
     * @brief Batch blind rotation
     * 
     * The main GPU kernel target. Each blind rotation is independent,
     * allowing massive parallelism.
     */
    virtual void BlindRotateBatch(
        const std::shared_ptr<RingGSWCryptoParams>& params,
        const std::vector<LWECiphertext>& cts,
        const RingGSWACCKey& ek,
        std::vector<RLWECiphertext>& accs
    ) = 0;
    
    /**
     * @brief Batch external product
     */
    virtual void ExternalProductBatch(
        const std::shared_ptr<RingGSWCryptoParams>& params,
        const std::vector<RingGSWEvalKey>& rgsws,
        const std::vector<RLWECiphertext>& rlwes,
        std::vector<RLWECiphertext>& results
    ) = 0;
    
    /**
     * @brief Batch key switching
     */
    virtual void KeySwitchBatch(
        const std::shared_ptr<LWECryptoParams>& params,
        const std::vector<LWECiphertext>& cts,
        const LWESwitchingKey& ks,
        std::vector<LWECiphertext>& results
    ) = 0;
    
    /**
     * @brief Batch modulus switching
     */
    virtual void ModSwitchBatch(
        const std::shared_ptr<LWECryptoParams>& params,
        const std::vector<LWECiphertext>& cts,
        std::vector<LWECiphertext>& results
    ) = 0;
    
    // ========================================================================
    // Packed Key/Ciphertext Formats
    // ========================================================================
    
    /**
     * @brief Pack bootstrapping key for device transfer
     * 
     * Converts the OpenFHE key structure to a packed format suitable
     * for GPU memory and kernel access patterns.
     */
    virtual DeviceBuffer PackBootstrappingKey(const RingGSWACCKey& ek) = 0;
    
    /**
     * @brief Unpack bootstrapping key from device format
     */
    virtual void UnpackBootstrappingKey(
        const DeviceBuffer& packed,
        RingGSWACCKey& ek
    ) = 0;
    
    /**
     * @brief Pack LWE ciphertexts for batch processing
     */
    virtual DeviceBuffer PackCiphertexts(
        const std::vector<LWECiphertext>& cts
    ) = 0;
    
    /**
     * @brief Unpack LWE ciphertexts from device format
     */
    virtual void UnpackCiphertexts(
        const DeviceBuffer& packed,
        std::vector<LWECiphertext>& cts
    ) = 0;
};

// ============================================================================
// CPU Backend - Wraps Existing GINX
// ============================================================================

/**
 * @brief CPU backend using existing OpenFHE GINX implementation
 * 
 * This is the correctness baseline. It simply delegates to the existing
 * OpenFHE code without modification.
 */
class BackendCPU : public Backend {
public:
    BackendCPU();
    ~BackendCPU() override;
    
    // Backend info
    BackendType Type() const override { return BackendType::CPU; }
    std::string Name() const override { return "CPU (OpenFHE GINX)"; }
    bool IsAvailable() const override { return true; }
    size_t MaxBatchSize() const override;
    size_t DeviceMemory() const override { return 0; }
    
    // Memory management (no-op for CPU)
    DeviceBuffer Allocate(size_t bytes) override;
    void Free(DeviceBuffer& buffer) override;
    void CopyToDevice(const void* host_ptr, DeviceBuffer& buffer, size_t bytes) override;
    void CopyToHost(const DeviceBuffer& buffer, void* host_ptr, size_t bytes) override;
    void Synchronize() override {}
    
    // Single operations (delegate to OpenFHE)
    void BlindRotate(
        const std::shared_ptr<RingGSWCryptoParams>& params,
        const LWECiphertext& ct,
        const RingGSWACCKey& ek,
        RLWECiphertext& acc
    ) override;
    
    void ExternalProduct(
        const std::shared_ptr<RingGSWCryptoParams>& params,
        const RingGSWEvalKey& rgsw,
        const RLWECiphertext& rlwe,
        RLWECiphertext& result
    ) override;
    
    void KeySwitch(
        const std::shared_ptr<LWECryptoParams>& params,
        const LWECiphertext& ct,
        const LWESwitchingKey& ks,
        LWECiphertext& result
    ) override;
    
    void ModSwitch(
        const std::shared_ptr<LWECryptoParams>& params,
        const LWECiphertext& ct,
        LWECiphertext& result
    ) override;
    
    // Batch operations (loop over single ops)
    void BlindRotateBatch(
        const std::shared_ptr<RingGSWCryptoParams>& params,
        const std::vector<LWECiphertext>& cts,
        const RingGSWACCKey& ek,
        std::vector<RLWECiphertext>& accs
    ) override;
    
    void ExternalProductBatch(
        const std::shared_ptr<RingGSWCryptoParams>& params,
        const std::vector<RingGSWEvalKey>& rgsws,
        const std::vector<RLWECiphertext>& rlwes,
        std::vector<RLWECiphertext>& results
    ) override;
    
    void KeySwitchBatch(
        const std::shared_ptr<LWECryptoParams>& params,
        const std::vector<LWECiphertext>& cts,
        const LWESwitchingKey& ks,
        std::vector<LWECiphertext>& results
    ) override;
    
    void ModSwitchBatch(
        const std::shared_ptr<LWECryptoParams>& params,
        const std::vector<LWECiphertext>& cts,
        std::vector<LWECiphertext>& results
    ) override;
    
    // Packed formats (identity transforms for CPU)
    DeviceBuffer PackBootstrappingKey(const RingGSWACCKey& ek) override;
    void UnpackBootstrappingKey(const DeviceBuffer& packed, RingGSWACCKey& ek) override;
    DeviceBuffer PackCiphertexts(const std::vector<LWECiphertext>& cts) override;
    void UnpackCiphertexts(const DeviceBuffer& packed, std::vector<LWECiphertext>& cts) override;
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// Backend Registry
// ============================================================================

/**
 * @brief Global backend registry and selection
 */
class BackendRegistry {
public:
    // Get singleton instance
    static BackendRegistry& Instance();
    
    // Register a backend
    void Register(std::unique_ptr<Backend> backend);
    
    // Get backend by type
    Backend* Get(BackendType type);
    
    // Get default backend (AUTO selection)
    Backend* GetDefault();
    
    // Set default backend type
    void SetDefault(BackendType type);
    
    // List available backends
    std::vector<BackendType> Available() const;
    
    // Check if a backend is available
    bool IsAvailable(BackendType type) const;
    
private:
    BackendRegistry();
    ~BackendRegistry();
    
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Convenience function to get current backend
inline Backend* CurrentBackend() {
    return BackendRegistry::Instance().GetDefault();
}

} // namespace backend
} // namespace lbcrypto

#endif // BACKEND_BACKEND_H
