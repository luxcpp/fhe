// =============================================================================
// Unified FHE NTT API - C Interface for Go CGO
// =============================================================================
//
// Single entry point for all platforms. MLX handles:
// - Metal (macOS/Apple Silicon) with native kernel optimization
// - CUDA (Linux/NVIDIA)
// - CPU fallback (all platforms)
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

#include "math/hal/mlx/mlx_backend.h"
#include <string>

#ifdef __APPLE__
#include "metal_dispatch_optimized.h"
#include <memory>
#include <unordered_map>
#include <mutex>

namespace lux {
namespace gpu {
namespace metal {

// Dispatcher cache for native Metal kernels
namespace {
    std::mutex g_cache_mutex;
    std::unordered_map<uint64_t, std::unique_ptr<NTTMetalDispatcherOptimized>> g_dispatcher_cache;

    uint64_t make_key(uint32_t N, uint64_t Q) {
        return (static_cast<uint64_t>(N) << 32) | (Q & 0xFFFFFFFF);
    }
}

NTTMetalDispatcherOptimized* get_dispatcher(uint32_t N, uint64_t Q) {
    uint64_t key = make_key(N, Q);
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    auto it = g_dispatcher_cache.find(key);
    if (it != g_dispatcher_cache.end()) return it->second.get();
    auto dispatcher = std::make_unique<NTTMetalDispatcherOptimized>(N, Q);
    auto* ptr = dispatcher.get();
    g_dispatcher_cache[key] = std::move(dispatcher);
    return ptr;
}

bool is_native_metal_available() {
    static bool checked = false, available = false;
    if (!checked) {
        try {
            NTTMetalDispatcherOptimized test(1024, 998244353);
            available = test.is_available();
        } catch (...) { available = false; }
        checked = true;
    }
    return available;
}

void clear_cache() {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    g_dispatcher_cache.clear();
}

} // namespace metal
} // namespace gpu
} // namespace lux
#endif // __APPLE__

// =============================================================================
// Unified C API - Works on ALL platforms
// =============================================================================

extern "C" {

bool fhe_gpu_available(void) {
#ifdef WITH_MLX
    return lux::mlx_backend::IsMLXAvailable();
#else
    return false;
#endif
}

const char* fhe_get_backend(void) {
#ifdef __APPLE__
    if (lux::gpu::metal::is_native_metal_available()) {
        return "Metal (Native)";
    }
#endif
#ifdef WITH_MLX
    if (lux::mlx_backend::IsMLXAvailable()) {
        static std::string device_name;
        device_name = lux::mlx_backend::GetDeviceName();
        return device_name.c_str();
    }
#endif
    return "CPU";
}

int fhe_ntt_forward(uint64_t* data, uint32_t N, uint64_t Q, uint32_t batch) {
#ifdef __APPLE__
    // Try native Metal first (13x faster)
    if (lux::gpu::metal::is_native_metal_available()) {
        try {
            auto* d = lux::gpu::metal::get_dispatcher(N, Q);
            if (d && d->is_available()) {
                d->forward(data, batch);
                return 0;
            }
        } catch (...) {}
    }
#endif
    // Fall back to MLX (handles Metal/CUDA/CPU)
#ifdef WITH_MLX
    return lux::mlx_backend::ntt_forward(data, N, Q, batch);
#else
    return -1;
#endif
}

int fhe_ntt_inverse(uint64_t* data, uint32_t N, uint64_t Q, uint32_t batch) {
#ifdef __APPLE__
    if (lux::gpu::metal::is_native_metal_available()) {
        try {
            auto* d = lux::gpu::metal::get_dispatcher(N, Q);
            if (d && d->is_available()) {
                d->inverse(data, batch);
                return 0;
            }
        } catch (...) {}
    }
#endif
#ifdef WITH_MLX
    return lux::mlx_backend::ntt_inverse(data, N, Q, batch);
#else
    return -1;
#endif
}

int fhe_pointwise_mul(uint64_t* result, const uint64_t* a, const uint64_t* b,
                      uint32_t N, uint64_t Q, uint32_t batch) {
#ifdef __APPLE__
    if (lux::gpu::metal::is_native_metal_available()) {
        try {
            auto* d = lux::gpu::metal::get_dispatcher(N, Q);
            if (d && d->is_available()) {
                d->pointwise_mul(result, a, b, batch);
                return 0;
            }
        } catch (...) {}
    }
#endif
#ifdef WITH_MLX
    return lux::mlx_backend::pointwise_mul(result, a, b, N, Q, batch);
#else
    return -1;
#endif
}

void fhe_clear_cache(void) {
#ifdef __APPLE__
    lux::gpu::metal::clear_cache();
#endif
#ifdef WITH_MLX
    lux::mlx_backend::clear_cache();
#endif
}

// Legacy Metal-specific API (for backwards compatibility)
bool metal_ntt_available(void) {
#ifdef __APPLE__
    return lux::gpu::metal::is_native_metal_available();
#else
    return false;
#endif
}

} // extern "C"
