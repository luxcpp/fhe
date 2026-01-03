// =============================================================================
// Native Metal NTT Wrapper - C++ Interface Header
// =============================================================================
//
// C++ interface to the native Metal NTT implementation.
// Use this header to access GPU-accelerated NTT operations.
//
// Example:
//   #include "metal_ntt_wrapper.h"
//   if (lux::gpu::metal::is_metal_ntt_available()) {
//       lux::gpu::metal::metal_ntt_forward(data, N, Q, batch);
//   }
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

#ifndef LUX_FHE_MATH_HAL_MLX_METAL_NTT_WRAPPER_H
#define LUX_FHE_MATH_HAL_MLX_METAL_NTT_WRAPPER_H

#include <cstdint>

namespace lux {
namespace gpu {
namespace metal {

// =============================================================================
// Availability Check
// =============================================================================

/// Check if native Metal NTT is available on this system.
/// Returns true on Apple Silicon with Metal support.
bool is_metal_ntt_available();

/// Check if the fused kernel is used for the given N.
/// Fused kernels are ~10x faster but limited to N <= 4096.
bool metal_ntt_uses_fused_kernel(uint32_t N, uint64_t Q);

// =============================================================================
// Synchronous NTT Operations
// =============================================================================

/// Forward NTT using native Metal GPU kernels.
/// @param data     In/out: Polynomial coefficients [batch * N]
/// @param N        Ring dimension (must be power of 2)
/// @param Q        Prime modulus
/// @param batch    Number of polynomials to transform
void metal_ntt_forward(uint64_t* data, uint32_t N, uint64_t Q, uint32_t batch = 1);

/// Inverse NTT using native Metal GPU kernels.
/// Includes scaling by N^{-1} mod Q.
void metal_ntt_inverse(uint64_t* data, uint32_t N, uint64_t Q, uint32_t batch = 1);

/// Pointwise multiplication: result[i] = a[i] * b[i] mod Q
void metal_pointwise_mul(uint64_t* result, const uint64_t* a, const uint64_t* b,
                          uint32_t N, uint64_t Q, uint32_t batch = 1);

// =============================================================================
// Asynchronous NTT Operations
// =============================================================================

/// Start async forward NTT. Use metal_ntt_wait_all to wait for completion.
void metal_ntt_forward_async(uint64_t* data, uint32_t N, uint64_t Q, uint32_t batch = 1);

/// Start async inverse NTT.
void metal_ntt_inverse_async(uint64_t* data, uint32_t N, uint64_t Q, uint32_t batch = 1);

/// Wait for all pending async operations for the given (N, Q) to complete.
void metal_ntt_wait_all(uint32_t N, uint64_t Q);

/// Check if there are pending async operations.
bool metal_ntt_has_pending(uint32_t N, uint64_t Q);

// =============================================================================
// Cache Management
// =============================================================================

/// Clear the dispatcher cache. Call at program exit or for testing.
void clear_metal_ntt_cache();

} // namespace metal
} // namespace gpu
} // namespace lux::fhe

#endif // LUX_FHE_MATH_HAL_MLX_METAL_NTT_WRAPPER_H
