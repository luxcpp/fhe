// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// Backend implementation - CPU backend wraps existing OpenFHE GINX

#include "backend/backend.h"
#include "rgsw-acc-cggi.h"
#include <stdexcept>
#include <algorithm>
#include <thread>

namespace lux::fhe {
namespace backend {

// ============================================================================
// BackendType utilities
// ============================================================================

std::string BackendTypeName(BackendType type) {
    switch (type) {
        case BackendType::CPU: return "CPU";
        case BackendType::MLX: return "MLX";
        case BackendType::CUDA: return "CUDA";
        case BackendType::AUTO: return "AUTO";
        default: return "UNKNOWN";
    }
}

BackendType BackendTypeFromName(const std::string& name) {
    if (name == "CPU" || name == "cpu") return BackendType::CPU;
    if (name == "MLX" || name == "mlx") return BackendType::MLX;
    if (name == "CUDA" || name == "cuda") {
        throw std::invalid_argument("CUDA backend requires enterprise license. Contact licensing@lux.network");
    }
    if (name == "AUTO" || name == "auto") return BackendType::AUTO;
    throw std::invalid_argument("Unknown backend type: " + name);
}

// ============================================================================
// BackendCPU Implementation
// ============================================================================

struct BackendCPU::Impl {
    // Reference to CGGI accumulator for GINX operations
    std::shared_ptr<RingGSWAccumulatorCGGI> cggi;
    
    Impl() : cggi(std::make_shared<RingGSWAccumulatorCGGI>()) {}
};

BackendCPU::BackendCPU() : impl_(std::make_unique<Impl>()) {}

BackendCPU::~BackendCPU() = default;

size_t BackendCPU::MaxBatchSize() const {
    // For CPU, we're limited by memory and threading
    // Return reasonable default based on hardware concurrency
    return std::max(1u, std::thread::hardware_concurrency()) * 64;
}

// Memory management - CPU uses host memory directly
DeviceBuffer BackendCPU::Allocate(size_t bytes) {
    DeviceBuffer buf;
    buf.ptr = std::malloc(bytes);
    buf.size = bytes;
    buf.device = BackendType::CPU;
    return buf;
}

void BackendCPU::Free(DeviceBuffer& buffer) {
    if (buffer.ptr && buffer.device == BackendType::CPU) {
        std::free(buffer.ptr);
        buffer.ptr = nullptr;
        buffer.size = 0;
    }
}

void BackendCPU::CopyToDevice(const void* host_ptr, DeviceBuffer& buffer, size_t bytes) {
    // For CPU, just memcpy
    if (buffer.ptr && bytes <= buffer.size) {
        std::memcpy(buffer.ptr, host_ptr, bytes);
    }
}

void BackendCPU::CopyToHost(const DeviceBuffer& buffer, void* host_ptr, size_t bytes) {
    // For CPU, just memcpy
    if (buffer.ptr && bytes <= buffer.size) {
        std::memcpy(host_ptr, buffer.ptr, bytes);
    }
}

// ============================================================================
// Core FHE Operations - Delegate to OpenFHE GINX
// ============================================================================

void BackendCPU::BlindRotate(
    const std::shared_ptr<RingGSWCryptoParams>& params,
    const LWECiphertext& ct,
    const RingGSWACCKey& ek,
    RLWECiphertext& acc
) {
    // TODO: Delegate to CGGI accumulator (GINX method)
    // The EvalAcc signature needs the LWE vector, not the ciphertext object
    // For now, this is a stub - proper implementation needs OpenFHE integration
    (void)params; (void)ct; (void)ek; (void)acc;
}

void BackendCPU::ExternalProduct(
    const std::shared_ptr<RingGSWCryptoParams>& params,
    const RingGSWEvalKey& rgsw,
    const RLWECiphertext& rlwe,
    RLWECiphertext& result
) {
    // TODO: Use CGGI's external product via public API
    // This is a stub - proper implementation needs OpenFHE integration
    (void)params; (void)rgsw; (void)rlwe;
    result = rlwe; // Placeholder - just copy input
}

void BackendCPU::KeySwitch(
    const std::shared_ptr<LWECryptoParams>& params,
    const LWECiphertext& ct,
    const LWESwitchingKey& ks,
    LWECiphertext& result
) {
    // OpenFHE's key switching is in the LWE scheme
    // This would delegate to LWEScheme::KeySwitch
    // For now, this is a stub that needs proper OpenFHE integration
    result = ct; // Placeholder
}

void BackendCPU::ModSwitch(
    const std::shared_ptr<LWECryptoParams>& params,
    const LWECiphertext& ct,
    LWECiphertext& result
) {
    // OpenFHE's modulus switching
    // This would delegate to LWEScheme::ModSwitch
    result = ct; // Placeholder
}

// ============================================================================
// Batch Operations - Loop over single ops (CPU baseline)
// ============================================================================

void BackendCPU::BlindRotateBatch(
    const std::shared_ptr<RingGSWCryptoParams>& params,
    const std::vector<LWECiphertext>& cts,
    const RingGSWACCKey& ek,
    std::vector<RLWECiphertext>& accs
) {
    accs.resize(cts.size());
    
    // Simple loop for CPU baseline
    // Could be parallelized with OpenMP if available
    #pragma omp parallel for if(cts.size() > 4)
    for (size_t i = 0; i < cts.size(); ++i) {
        BlindRotate(params, cts[i], ek, accs[i]);
    }
}

void BackendCPU::ExternalProductBatch(
    const std::shared_ptr<RingGSWCryptoParams>& params,
    const std::vector<RingGSWEvalKey>& rgsws,
    const std::vector<RLWECiphertext>& rlwes,
    std::vector<RLWECiphertext>& results
) {
    if (rgsws.size() != rlwes.size()) {
        throw std::invalid_argument("Batch size mismatch in ExternalProductBatch");
    }

    results.resize(rgsws.size());

    #pragma omp parallel for if(rgsws.size() > 4)
    for (size_t i = 0; i < rgsws.size(); ++i) {
        ExternalProduct(params, rgsws[i], rlwes[i], results[i]);
    }
}

void BackendCPU::KeySwitchBatch(
    const std::shared_ptr<LWECryptoParams>& params,
    const std::vector<LWECiphertext>& cts,
    const LWESwitchingKey& ks,
    std::vector<LWECiphertext>& results
) {
    results.resize(cts.size());
    
    #pragma omp parallel for if(cts.size() > 4)
    for (size_t i = 0; i < cts.size(); ++i) {
        KeySwitch(params, cts[i], ks, results[i]);
    }
}

void BackendCPU::ModSwitchBatch(
    const std::shared_ptr<LWECryptoParams>& params,
    const std::vector<LWECiphertext>& cts,
    std::vector<LWECiphertext>& results
) {
    results.resize(cts.size());
    
    #pragma omp parallel for if(cts.size() > 4)
    for (size_t i = 0; i < cts.size(); ++i) {
        ModSwitch(params, cts[i], results[i]);
    }
}

// ============================================================================
// Packed Formats - Identity for CPU
// ============================================================================

DeviceBuffer BackendCPU::PackBootstrappingKey(const RingGSWACCKey& ek) {
    // For CPU, we don't need special packing
    // Just store a pointer to the key (not ownership)
    DeviceBuffer buf;
    buf.ptr = const_cast<void*>(static_cast<const void*>(&ek));
    buf.size = sizeof(ek);
    buf.device = BackendType::CPU;
    return buf;
}

void BackendCPU::UnpackBootstrappingKey(const DeviceBuffer& packed, RingGSWACCKey& ek) {
    // For CPU, the key is already in usable form
    // This is a no-op for the baseline
}

DeviceBuffer BackendCPU::PackCiphertexts(const std::vector<LWECiphertext>& cts) {
    DeviceBuffer buf;
    buf.ptr = const_cast<void*>(static_cast<const void*>(cts.data()));
    buf.size = cts.size() * sizeof(LWECiphertext);
    buf.device = BackendType::CPU;
    return buf;
}

void BackendCPU::UnpackCiphertexts(const DeviceBuffer& packed, std::vector<LWECiphertext>& cts) {
    // For CPU, ciphertexts are already in usable form
}

// ============================================================================
// BackendRegistry Implementation
// ============================================================================

struct BackendRegistry::Impl {
    std::map<BackendType, std::unique_ptr<Backend>> backends;
    BackendType default_type = BackendType::CPU;
    std::mutex mutex;
    
    Impl() {
        // Register CPU backend by default
        backends[BackendType::CPU] = std::make_unique<BackendCPU>();
    }
};

BackendRegistry& BackendRegistry::Instance() {
    static BackendRegistry instance;
    return instance;
}

BackendRegistry::BackendRegistry() : impl_(std::make_unique<Impl>()) {}

BackendRegistry::~BackendRegistry() = default;

void BackendRegistry::Register(std::unique_ptr<Backend> backend) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->backends[backend->Type()] = std::move(backend);
}

Backend* BackendRegistry::Get(BackendType type) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto it = impl_->backends.find(type);
    if (it != impl_->backends.end()) {
        return it->second.get();
    }
    return nullptr;
}

Backend* BackendRegistry::GetDefault() {
    if (impl_->default_type == BackendType::AUTO) {
        // Auto-select: prefer MLX (Apple Silicon) if available, fall back to CPU
        // CUDA requires enterprise license - contact licensing@lux.network
        if (auto* mlx = Get(BackendType::MLX); mlx && mlx->IsAvailable()) {
            return mlx;
        }
        return Get(BackendType::CPU);
    }
    return Get(impl_->default_type);
}

void BackendRegistry::SetDefault(BackendType type) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->default_type = type;
}

std::vector<BackendType> BackendRegistry::Available() const {
    std::vector<BackendType> result;
    for (const auto& [type, backend] : impl_->backends) {
        if (backend && backend->IsAvailable()) {
            result.push_back(type);
        }
    }
    return result;
}

bool BackendRegistry::IsAvailable(BackendType type) const {
    auto it = impl_->backends.find(type);
    return it != impl_->backends.end() && it->second && it->second->IsAvailable();
}

} // namespace backend
} // namespace lux::fhe
