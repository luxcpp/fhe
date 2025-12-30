// =============================================================================
// Barrett Reduction for Metal GPU - 32-bit RNS System
// =============================================================================
//
// Constant-time Barrett reduction without 128-bit arithmetic.
// Designed for FHE operations on Apple Metal GPU.
//
// Key design decisions:
// 1. 32-bit limbs: Metal lacks native 128-bit; 32x32->64 fits in uint64_t
// 2. Fused multiply-reduce: (a * b) mod Q in single operation
// 3. Precomputed constants: mu, Q, etc. computed on CPU
// 4. Constant-time: No branching on secret data for security
// 5. Batch operations: [batch, N] tensor layout
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

#ifndef LBCRYPTO_MATH_HAL_MLX_BARRETT_METAL_H
#define LBCRYPTO_MATH_HAL_MLX_BARRETT_METAL_H

#include <cstdint>
#include <vector>
#include <stdexcept>
#include <algorithm>

#ifdef WITH_MLX
#include <mlx/mlx.h>
namespace mx = mlx::core;
#endif

namespace lbcrypto {
namespace gpu {
namespace metal {

// =============================================================================
// Barrett Parameters for 32-bit Modulus
// =============================================================================
//
// For modulus Q < 2^31:
//   mu = floor(2^64 / Q)  -- Barrett constant
//   k = 64                 -- Reduction shift
//
// Barrett reduction: a * b mod Q
//   1. p = a * b           (fits in 64 bits since a,b < 2^32)
//   2. q = mulhi(p, mu)    (high 64 bits of p * mu, approximates p/Q)
//   3. r = p - q * Q       (remainder)
//   4. if r >= Q: r -= Q   (final correction)

struct Barrett32Params {
    uint32_t Q;           // Prime modulus (< 2^31 for safety)
    uint64_t mu;          // Barrett constant: floor(2^64 / Q)
    uint32_t Q_neg;       // -Q mod 2^32 (for subtraction optimization)
    uint32_t N_inv;       // N^{-1} mod Q for inverse NTT scaling
    uint64_t N_inv_mu;    // Precomputed (N_inv << 32) for faster scaling

    // Create parameters for given modulus
    static Barrett32Params create(uint32_t Q, uint32_t N = 0) {
        if (Q == 0 || (Q & (Q - 1)) == 0) {
            throw std::invalid_argument("Q must be prime (non-zero, non-power-of-2)");
        }
        if (Q >= (1u << 31)) {
            throw std::invalid_argument("Q must be < 2^31 for Barrett32");
        }

        Barrett32Params p;
        p.Q = Q;

        // mu = floor(2^64 / Q)
        // Compute as (2^64 - 1) / Q + adjustment
        __uint128_t pow2_64 = static_cast<__uint128_t>(1) << 64;
        p.mu = static_cast<uint64_t>(pow2_64 / Q);

        // -Q mod 2^32
        p.Q_neg = static_cast<uint32_t>(-static_cast<int64_t>(Q));

        // Compute N^{-1} mod Q if N provided
        if (N > 0) {
            // Extended Euclidean algorithm
            int64_t t = 0, newt = 1;
            int64_t r = Q, newr = N;
            while (newr != 0) {
                int64_t quotient = r / newr;
                std::tie(t, newt) = std::make_pair(newt, t - quotient * newt);
                std::tie(r, newr) = std::make_pair(newr, r - quotient * newr);
            }
            p.N_inv = static_cast<uint32_t>((t < 0) ? t + Q : t);
            p.N_inv_mu = static_cast<uint64_t>(p.N_inv) << 32;
        } else {
            p.N_inv = 1;
            p.N_inv_mu = static_cast<uint64_t>(1) << 32;
        }

        return p;
    }
};

// =============================================================================
// CPU Reference Implementation (for verification)
// =============================================================================

// Barrett multiply-reduce: (a * b) mod Q
// Constant-time: no early exits or data-dependent branches
inline uint32_t barrett_mul_mod_cpu(uint32_t a, uint32_t b,
                                     uint32_t Q, uint64_t mu) {
    // Step 1: Compute product (max 63 bits for a,b < 2^31)
    uint64_t p = static_cast<uint64_t>(a) * b;

    // Step 2: Approximate quotient via Barrett
    // q = floor((p * mu) / 2^64) = mulhi(p, mu)
    __uint128_t pm = static_cast<__uint128_t>(p) * mu;
    uint64_t q = static_cast<uint64_t>(pm >> 64);

    // Step 3: Compute remainder
    uint64_t r = p - q * Q;

    // Step 4: Constant-time final reduction
    // Use conditional select instead of branch
    // If r >= Q, subtract Q; result is in [0, Q)
    uint32_t r32 = static_cast<uint32_t>(r);
    uint32_t mask = static_cast<uint32_t>(-static_cast<int32_t>(r32 >= Q));
    return r32 - (mask & Q);
}

// Modular addition: (a + b) mod Q
// Constant-time implementation
inline uint32_t mod_add_cpu(uint32_t a, uint32_t b, uint32_t Q) {
    uint32_t sum = a + b;
    uint32_t mask = static_cast<uint32_t>(-static_cast<int32_t>(sum >= Q));
    return sum - (mask & Q);
}

// Modular subtraction: (a - b) mod Q
// Constant-time implementation
inline uint32_t mod_sub_cpu(uint32_t a, uint32_t b, uint32_t Q) {
    uint32_t diff = a - b;
    uint32_t mask = static_cast<uint32_t>(-static_cast<int32_t>(a < b));
    return diff + (mask & Q);
}

// =============================================================================
// Metal Shader Source for Barrett Reduction
// =============================================================================

inline const char* get_barrett_metal_source() {
    return R"METAL(
// =============================================================================
// Barrett Reduction Kernels for 32-bit RNS - Apple Metal
// =============================================================================
//
// Security: Constant-time implementation (no branching on secret data)
// Performance: Uses Metal's mulhi for high bits of 64-bit multiply

#include <metal_stdlib>
using namespace metal;

// Barrett parameters (passed via buffer)
struct Barrett32Params {
    uint32_t Q;           // Prime modulus
    uint64_t mu;          // Barrett constant: floor(2^64 / Q)
    uint32_t Q_neg;       // -Q mod 2^32
    uint32_t N_inv;       // N^{-1} mod Q
    uint64_t N_inv_mu;    // Precomputed for scaling
};

// =============================================================================
// Core Barrett Multiply-Reduce (Constant-Time)
// =============================================================================
//
// Computes (a * b) mod Q without division
// Uses Barrett reduction: r = a*b - floor((a*b * mu) >> 64) * Q

inline uint32_t barrett_mul_mod(uint32_t a, uint32_t b,
                                 uint32_t Q, uint64_t mu) {
    // Product fits in 64 bits (32x32)
    uint64_t p = uint64_t(a) * uint64_t(b);

    // High 64 bits of p * mu gives approximate quotient
    uint64_t q = metal::mulhi(p, mu);

    // Compute remainder
    uint64_t r = p - q * uint64_t(Q);

    // Constant-time final reduction using select (no branch)
    // If r >= Q, we need r - Q; otherwise r
    // Use mask: if r >= Q, mask = 0xFFFFFFFF, else 0
    uint32_t r32 = uint32_t(r);
    uint32_t ge_mask = uint32_t(int32_t(r32 >= Q) * -1);
    return r32 - (ge_mask & Q);
}

// Alternative: Two-step reduction for values up to 2Q
inline uint32_t barrett_reduce_single(uint64_t x, uint32_t Q, uint64_t mu) {
    uint64_t q = metal::mulhi(x, mu);
    uint64_t r = x - q * uint64_t(Q);
    uint32_t r32 = uint32_t(r);
    uint32_t ge_mask = uint32_t(int32_t(r32 >= Q) * -1);
    return r32 - (ge_mask & Q);
}

// =============================================================================
// Modular Addition (Constant-Time)
// =============================================================================

inline uint32_t mod_add(uint32_t a, uint32_t b, uint32_t Q) {
    uint32_t sum = a + b;
    uint32_t ge_mask = uint32_t(int32_t(sum >= Q) * -1);
    return sum - (ge_mask & Q);
}

// =============================================================================
// Modular Subtraction (Constant-Time)
// =============================================================================

inline uint32_t mod_sub(uint32_t a, uint32_t b, uint32_t Q) {
    uint32_t diff = a - b;
    uint32_t lt_mask = uint32_t(int32_t(a < b) * -1);
    return diff + (lt_mask & Q);
}

// =============================================================================
// Batch Barrett Multiply-Reduce Kernel
// =============================================================================
//
// Computes result[i] = (a[i] * b[i]) mod Q for all i
// Layout: [batch, N] where each thread handles one coefficient

kernel void barrett_mul_mod_batch(
    device uint32_t* result          [[buffer(0)]],
    constant uint32_t* a             [[buffer(1)]],
    constant uint32_t* b             [[buffer(2)]],
    constant Barrett32Params& params [[buffer(3)]],
    constant uint32_t& size          [[buffer(4)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= size) return;

    result[gid] = barrett_mul_mod(a[gid], b[gid], params.Q, params.mu);
}

// =============================================================================
// Batch Modular Add Kernel
// =============================================================================

kernel void mod_add_batch(
    device uint32_t* result          [[buffer(0)]],
    constant uint32_t* a             [[buffer(1)]],
    constant uint32_t* b             [[buffer(2)]],
    constant Barrett32Params& params [[buffer(3)]],
    constant uint32_t& size          [[buffer(4)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= size) return;

    result[gid] = mod_add(a[gid], b[gid], params.Q);
}

// =============================================================================
// Batch Modular Sub Kernel
// =============================================================================

kernel void mod_sub_batch(
    device uint32_t* result          [[buffer(0)]],
    constant uint32_t* a             [[buffer(1)]],
    constant uint32_t* b             [[buffer(2)]],
    constant Barrett32Params& params [[buffer(3)]],
    constant uint32_t& size          [[buffer(4)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= size) return;

    result[gid] = mod_sub(a[gid], b[gid], params.Q);
}

// =============================================================================
// Fused Multiply-Add-Reduce: result = (a * b + c) mod Q
// =============================================================================
//
// Useful for accumulating products in NTT butterfly

kernel void barrett_fma_mod_batch(
    device uint32_t* result          [[buffer(0)]],
    constant uint32_t* a             [[buffer(1)]],
    constant uint32_t* b             [[buffer(2)]],
    constant uint32_t* c             [[buffer(3)]],
    constant Barrett32Params& params [[buffer(4)]],
    constant uint32_t& size          [[buffer(5)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= size) return;

    uint32_t Q = params.Q;
    uint64_t mu = params.mu;

    // Compute (a * b) mod Q
    uint32_t ab_mod = barrett_mul_mod(a[gid], b[gid], Q, mu);

    // Add c and reduce
    result[gid] = mod_add(ab_mod, c[gid], Q);
}

// =============================================================================
// NTT Butterfly with Barrett Reduction
// =============================================================================
//
// Cooley-Tukey butterfly: (lo, hi) -> (lo + w*hi, lo - w*hi) mod Q
// Used in forward NTT

kernel void ntt_butterfly_ct(
    device uint32_t* data            [[buffer(0)]],
    constant uint32_t* twiddles      [[buffer(1)]],
    constant Barrett32Params& params [[buffer(2)]],
    constant uint32_t& N             [[buffer(3)]],
    constant uint32_t& stage         [[buffer(4)]],
    constant uint32_t& batch_size    [[buffer(5)]],
    uint2 gid                        [[thread_position_in_grid]]
) {
    uint32_t butterfly_idx = gid.x;
    uint32_t batch_idx = gid.y;

    uint32_t num_butterflies = N >> 1;
    if (butterfly_idx >= num_butterflies || batch_idx >= batch_size) return;

    uint32_t Q = params.Q;
    uint64_t mu = params.mu;

    // Stage s: m = 2^s groups, t = N/(2^{s+1}) butterflies per group
    uint32_t m = 1u << stage;
    uint32_t t = N >> (stage + 1);

    // Map butterfly index to data indices
    uint32_t i = butterfly_idx / t;
    uint32_t j = butterfly_idx % t;

    uint32_t log_N = 0;
    for (uint32_t n = N; n > 1; n >>= 1) log_N++;

    uint32_t idx_lo = (i << (log_N - stage)) + j;
    uint32_t idx_hi = idx_lo + t;

    device uint32_t* poly = data + batch_idx * N;

    // Load values
    uint32_t lo_val = poly[idx_lo];
    uint32_t hi_val = poly[idx_hi];
    uint32_t w = twiddles[m + i];

    // Compute w * hi mod Q
    uint32_t whi = barrett_mul_mod(hi_val, w, Q, mu);

    // Butterfly
    poly[idx_lo] = mod_add(lo_val, whi, Q);
    poly[idx_hi] = mod_sub(lo_val, whi, Q);
}

// =============================================================================
// NTT Butterfly (Gentleman-Sande) with Barrett Reduction
// =============================================================================
//
// GS butterfly: (lo, hi) -> (lo + hi, (lo - hi) * w) mod Q
// Used in inverse NTT

kernel void ntt_butterfly_gs(
    device uint32_t* data            [[buffer(0)]],
    constant uint32_t* inv_twiddles  [[buffer(1)]],
    constant Barrett32Params& params [[buffer(2)]],
    constant uint32_t& N             [[buffer(3)]],
    constant uint32_t& stage         [[buffer(4)]],
    constant uint32_t& batch_size    [[buffer(5)]],
    uint2 gid                        [[thread_position_in_grid]]
) {
    uint32_t butterfly_idx = gid.x;
    uint32_t batch_idx = gid.y;

    uint32_t num_butterflies = N >> 1;
    if (butterfly_idx >= num_butterflies || batch_idx >= batch_size) return;

    uint32_t Q = params.Q;
    uint64_t mu = params.mu;

    // Stage s: m = N/2^{s+1} groups, t = 2^s butterflies per group
    uint32_t m = N >> (stage + 1);
    uint32_t t = 1u << stage;

    // Map butterfly index to data indices
    uint32_t i = butterfly_idx / t;
    uint32_t j = butterfly_idx % t;

    uint32_t idx_lo = (i << (stage + 1)) + j;
    uint32_t idx_hi = idx_lo + t;

    device uint32_t* poly = data + batch_idx * N;

    // Load values
    uint32_t lo_val = poly[idx_lo];
    uint32_t hi_val = poly[idx_hi];
    uint32_t w = inv_twiddles[m + i];

    // GS butterfly
    uint32_t sum = mod_add(lo_val, hi_val, Q);
    uint32_t diff = mod_sub(lo_val, hi_val, Q);
    uint32_t diff_w = barrett_mul_mod(diff, w, Q, mu);

    poly[idx_lo] = sum;
    poly[idx_hi] = diff_w;
}

// =============================================================================
// Scale by N^{-1} (Final step of inverse NTT)
// =============================================================================

kernel void ntt_scale_n_inv(
    device uint32_t* data            [[buffer(0)]],
    constant Barrett32Params& params [[buffer(1)]],
    constant uint32_t& N             [[buffer(2)]],
    constant uint32_t& batch_size    [[buffer(3)]],
    uint2 gid                        [[thread_position_in_grid]]
) {
    uint32_t coeff_idx = gid.x;
    uint32_t batch_idx = gid.y;

    if (coeff_idx >= N || batch_idx >= batch_size) return;

    device uint32_t* poly = data + batch_idx * N;

    poly[coeff_idx] = barrett_mul_mod(
        poly[coeff_idx],
        params.N_inv,
        params.Q,
        params.mu
    );
}

// =============================================================================
// Pointwise Multiply-Accumulate for External Product
// =============================================================================
//
// Computes: acc[i] = (acc[i] + a[i] * b[i]) mod Q
// Used in RGSW external product

kernel void pointwise_mac(
    device uint32_t* acc             [[buffer(0)]],
    constant uint32_t* a             [[buffer(1)]],
    constant uint32_t* b             [[buffer(2)]],
    constant Barrett32Params& params [[buffer(3)]],
    constant uint32_t& size          [[buffer(4)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= size) return;

    uint32_t Q = params.Q;
    uint64_t mu = params.mu;

    uint32_t prod = barrett_mul_mod(a[gid], b[gid], Q, mu);
    acc[gid] = mod_add(acc[gid], prod, Q);
}

// =============================================================================
// Threaded NTT with Shared Memory (For N <= 1024)
// =============================================================================
//
// Complete forward NTT in shared memory with threadgroup barriers

kernel void ntt_forward_complete_32bit(
    device uint32_t* data            [[buffer(0)]],
    constant uint32_t* twiddles      [[buffer(1)]],
    constant Barrett32Params& params [[buffer(2)]],
    constant uint32_t& N             [[buffer(3)]],
    constant uint32_t& log_N         [[buffer(4)]],
    constant uint32_t& batch_size    [[buffer(5)]],
    uint2 tid                        [[thread_position_in_grid]],
    uint2 tg_size                    [[threads_per_threadgroup]],
    uint2 tg_id                      [[threadgroup_position_in_grid]],
    threadgroup uint32_t* shared     [[threadgroup(0)]]
) {
    uint32_t batch_idx = tg_id.y;
    uint32_t local_idx = tid.x % tg_size.x;

    if (batch_idx >= batch_size) return;

    uint32_t Q = params.Q;
    uint64_t mu = params.mu;

    device uint32_t* poly = data + batch_idx * N;

    // Load to shared memory
    for (uint32_t i = local_idx; i < N; i += tg_size.x) {
        shared[i] = poly[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cooley-Tukey stages
    for (uint32_t stage = 0; stage < log_N; ++stage) {
        uint32_t m = 1u << stage;
        uint32_t t = N >> (stage + 1);

        for (uint32_t butterfly_idx = local_idx; butterfly_idx < N/2; butterfly_idx += tg_size.x) {
            uint32_t i = butterfly_idx / t;
            uint32_t j = butterfly_idx % t;

            uint32_t idx_lo = (i << (log_N - stage)) + j;
            uint32_t idx_hi = idx_lo + t;

            uint32_t tw_idx = m + i;
            uint32_t w = twiddles[tw_idx];

            uint32_t lo_val = shared[idx_lo];
            uint32_t hi_val = shared[idx_hi];

            uint32_t whi = barrett_mul_mod(hi_val, w, Q, mu);

            shared[idx_lo] = mod_add(lo_val, whi, Q);
            shared[idx_hi] = mod_sub(lo_val, whi, Q);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back
    for (uint32_t i = local_idx; i < N; i += tg_size.x) {
        poly[i] = shared[i];
    }
}

// =============================================================================
// Complete Inverse NTT with Shared Memory
// =============================================================================

kernel void ntt_inverse_complete_32bit(
    device uint32_t* data            [[buffer(0)]],
    constant uint32_t* inv_twiddles  [[buffer(1)]],
    constant Barrett32Params& params [[buffer(2)]],
    constant uint32_t& N             [[buffer(3)]],
    constant uint32_t& log_N         [[buffer(4)]],
    constant uint32_t& batch_size    [[buffer(5)]],
    uint2 tid                        [[thread_position_in_grid]],
    uint2 tg_size                    [[threads_per_threadgroup]],
    uint2 tg_id                      [[threadgroup_position_in_grid]],
    threadgroup uint32_t* shared     [[threadgroup(0)]]
) {
    uint32_t batch_idx = tg_id.y;
    uint32_t local_idx = tid.x % tg_size.x;

    if (batch_idx >= batch_size) return;

    uint32_t Q = params.Q;
    uint64_t mu = params.mu;
    uint32_t N_inv = params.N_inv;

    device uint32_t* poly = data + batch_idx * N;

    // Load to shared memory
    for (uint32_t i = local_idx; i < N; i += tg_size.x) {
        shared[i] = poly[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Gentleman-Sande stages
    for (uint32_t stage = 0; stage < log_N; ++stage) {
        uint32_t m = N >> (stage + 1);
        uint32_t t = 1u << stage;

        for (uint32_t butterfly_idx = local_idx; butterfly_idx < N/2; butterfly_idx += tg_size.x) {
            uint32_t i = butterfly_idx / t;
            uint32_t j = butterfly_idx % t;

            uint32_t idx_lo = (i << (stage + 1)) + j;
            uint32_t idx_hi = idx_lo + t;

            uint32_t tw_idx = m + i;
            uint32_t w = inv_twiddles[tw_idx];

            uint32_t lo_val = shared[idx_lo];
            uint32_t hi_val = shared[idx_hi];

            uint32_t sum = mod_add(lo_val, hi_val, Q);
            uint32_t diff = mod_sub(lo_val, hi_val, Q);
            uint32_t diff_w = barrett_mul_mod(diff, w, Q, mu);

            shared[idx_lo] = sum;
            shared[idx_hi] = diff_w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Scale by N^{-1} and write back
    for (uint32_t i = local_idx; i < N; i += tg_size.x) {
        poly[i] = barrett_mul_mod(shared[i], N_inv, Q, mu);
    }
}

)METAL";
}

#ifdef WITH_MLX

// =============================================================================
// Barrett32 Dispatcher for MLX
// =============================================================================
//
// Provides high-level API for Barrett reduction operations on GPU

class Barrett32Dispatcher {
public:
    explicit Barrett32Dispatcher(uint32_t Q, uint32_t N = 0);

    // Core operations
    mx::array mul_mod(const mx::array& a, const mx::array& b);
    mx::array add_mod(const mx::array& a, const mx::array& b);
    mx::array sub_mod(const mx::array& a, const mx::array& b);
    mx::array fma_mod(const mx::array& a, const mx::array& b, const mx::array& c);

    // In-place accumulate: acc += a * b mod Q
    void mac_inplace(mx::array& acc, const mx::array& a, const mx::array& b);

    // NTT operations using Barrett reduction
    void ntt_forward(mx::array& data);
    void ntt_inverse(mx::array& data);

    // Pointwise multiply in NTT domain
    mx::array pointwise_mul(const mx::array& a, const mx::array& b);

    // Full polynomial multiply: INTT(NTT(a) * NTT(b))
    mx::array poly_mul(const mx::array& a, const mx::array& b);

    // Accessors
    const Barrett32Params& params() const { return params_; }
    uint32_t modulus() const { return params_.Q; }
    bool is_available() const { return available_; }

private:
    Barrett32Params params_;
    uint32_t N_ = 0;
    uint32_t log_N_ = 0;
    bool available_ = false;

    // Twiddle factors (32-bit) - use shared_ptr to avoid default constructor issues
    std::vector<uint32_t> twiddles_;
    std::vector<uint32_t> inv_twiddles_;
    std::shared_ptr<mx::array> tw_gpu_;
    std::shared_ptr<mx::array> inv_tw_gpu_;

    void init_twiddles();

    // CPU fallback implementations
    void ntt_forward_cpu(std::vector<uint32_t>& data);
    void ntt_inverse_cpu(std::vector<uint32_t>& data);
};

// =============================================================================
// Implementation
// =============================================================================

inline Barrett32Dispatcher::Barrett32Dispatcher(uint32_t Q, uint32_t N)
    : params_(Barrett32Params::create(Q, N)), N_(N) {

    available_ = mx::metal::is_available();

    if (N > 0) {
        log_N_ = 0;
        while ((1u << log_N_) < N) ++log_N_;

        if ((1u << log_N_) != N) {
            throw std::invalid_argument("N must be a power of 2");
        }

        init_twiddles();
    }

    if (available_) {
        mx::set_default_device(mx::Device::gpu);
    }
}

inline void Barrett32Dispatcher::init_twiddles() {
    if (N_ == 0) return;

    uint32_t Q = params_.Q;

    // Find primitive 2N-th root of unity
    auto powmod32 = [](uint32_t base, uint32_t exp, uint32_t m) -> uint32_t {
        uint64_t result = 1, b = base;
        while (exp > 0) {
            if (exp & 1) result = (result * b) % m;
            exp >>= 1;
            b = (b * b) % m;
        }
        return static_cast<uint32_t>(result);
    };

    auto mod_inv32 = [](uint32_t a, uint32_t m) -> uint32_t {
        int64_t t = 0, newt = 1;
        int64_t r = m, newr = a;
        while (newr != 0) {
            int64_t q = r / newr;
            std::tie(t, newt) = std::make_pair(newt, t - q * newt);
            std::tie(r, newr) = std::make_pair(newr, r - q * newr);
        }
        return static_cast<uint32_t>((t < 0) ? t + m : t);
    };

    // Find generator and omega
    uint32_t omega = 0;
    uint32_t omega_inv = 0;

    for (uint32_t g = 2; g < Q; ++g) {
        if (powmod32(g, (Q - 1) / 2, Q) != 1) {
            omega = powmod32(g, (Q - 1) / (2 * N_), Q);
            omega_inv = mod_inv32(omega, Q);
            break;
        }
    }

    if (omega == 0) {
        throw std::runtime_error("Could not find primitive root");
    }

    // Bit-reverse helper
    auto bit_reverse = [this](uint32_t x) -> uint32_t {
        uint32_t result = 0;
        for (uint32_t i = 0; i < log_N_; ++i) {
            result = (result << 1) | (x & 1);
            x >>= 1;
        }
        return result;
    };

    // Compute twiddles in bit-reversed order (OpenFHE compatible)
    twiddles_.resize(N_);
    inv_twiddles_.resize(N_);

    for (uint32_t m = 1; m < N_; m <<= 1) {
        uint32_t log_m = 0;
        while ((1u << log_m) < m) ++log_m;

        for (uint32_t i = 0; i < m; ++i) {
            uint32_t exp = (N_ / m) * bit_reverse(i);
            twiddles_[m + i] = powmod32(omega, exp, Q);
            inv_twiddles_[m + i] = powmod32(omega_inv, exp, Q);
        }
    }
    twiddles_[0] = 1;
    inv_twiddles_[0] = 1;

    // Upload to GPU (using shared_ptr)
    tw_gpu_ = std::make_shared<mx::array>(
        mx::array(reinterpret_cast<int32_t*>(twiddles_.data()),
                  {static_cast<int>(N_)}, mx::int32));
    inv_tw_gpu_ = std::make_shared<mx::array>(
        mx::array(reinterpret_cast<int32_t*>(inv_twiddles_.data()),
                  {static_cast<int>(N_)}, mx::int32));
    mx::eval(*tw_gpu_);
    mx::eval(*inv_tw_gpu_);
}

inline mx::array Barrett32Dispatcher::mul_mod(const mx::array& a, const mx::array& b) {
    // For now, use CPU-side Barrett reduction via MLX ops
    // (Full Metal kernel would require custom op registration)

    auto a64 = mx::astype(a, mx::int64);
    auto b64 = mx::astype(b, mx::int64);
    auto Q64 = mx::array(static_cast<int64_t>(params_.Q));

    auto prod = mx::multiply(a64, b64);
    auto result = mx::remainder(prod, Q64);

    return mx::astype(result, mx::int32);
}

inline mx::array Barrett32Dispatcher::add_mod(const mx::array& a, const mx::array& b) {
    auto a64 = mx::astype(a, mx::int64);
    auto b64 = mx::astype(b, mx::int64);
    auto Q64 = mx::array(static_cast<int64_t>(params_.Q));

    auto sum = mx::add(a64, b64);
    auto result = mx::remainder(sum, Q64);

    return mx::astype(result, mx::int32);
}

inline mx::array Barrett32Dispatcher::sub_mod(const mx::array& a, const mx::array& b) {
    auto a64 = mx::astype(a, mx::int64);
    auto b64 = mx::astype(b, mx::int64);
    auto Q64 = mx::array(static_cast<int64_t>(params_.Q));

    auto diff = mx::add(mx::subtract(a64, b64), Q64);
    auto result = mx::remainder(diff, Q64);

    return mx::astype(result, mx::int32);
}

inline mx::array Barrett32Dispatcher::fma_mod(const mx::array& a, const mx::array& b,
                                               const mx::array& c) {
    auto ab = mul_mod(a, b);
    return add_mod(ab, c);
}

inline void Barrett32Dispatcher::mac_inplace(mx::array& acc,
                                              const mx::array& a,
                                              const mx::array& b) {
    auto prod = mul_mod(a, b);
    acc = add_mod(acc, prod);
    mx::eval(acc);
}

inline void Barrett32Dispatcher::ntt_forward_cpu(std::vector<uint32_t>& data) {
    uint32_t Q = params_.Q;
    uint64_t mu = params_.mu;

    for (uint32_t s = 0; s < log_N_; ++s) {
        uint32_t m = 1u << s;
        uint32_t t = N_ >> (s + 1);

        for (uint32_t i = 0; i < m; ++i) {
            uint32_t j1 = i << (log_N_ - s);
            uint32_t j2 = j1 + t;
            uint32_t w = twiddles_[m + i];

            for (uint32_t j = j1; j < j2; ++j) {
                uint32_t lo = data[j];
                uint32_t hi = data[j + t];
                uint32_t whi = barrett_mul_mod_cpu(hi, w, Q, mu);
                data[j] = mod_add_cpu(lo, whi, Q);
                data[j + t] = mod_sub_cpu(lo, whi, Q);
            }
        }
    }
}

inline void Barrett32Dispatcher::ntt_inverse_cpu(std::vector<uint32_t>& data) {
    uint32_t Q = params_.Q;
    uint64_t mu = params_.mu;

    for (uint32_t s = 0; s < log_N_; ++s) {
        uint32_t m = N_ >> (s + 1);
        uint32_t t = 1u << s;

        for (uint32_t i = 0; i < m; ++i) {
            uint32_t j1 = i << (s + 1);
            uint32_t j2 = j1 + t;
            uint32_t w = inv_twiddles_[m + i];

            for (uint32_t j = j1; j < j2; ++j) {
                uint32_t lo = data[j];
                uint32_t hi = data[j + t];
                data[j] = mod_add_cpu(lo, hi, Q);
                uint32_t diff = mod_sub_cpu(lo, hi, Q);
                data[j + t] = barrett_mul_mod_cpu(diff, w, Q, mu);
            }
        }
    }

    // Scale by N^{-1}
    for (uint32_t i = 0; i < N_; ++i) {
        data[i] = barrett_mul_mod_cpu(data[i], params_.N_inv, Q, mu);
    }
}

inline void Barrett32Dispatcher::ntt_forward(mx::array& data) {
    auto shape = data.shape();
    int batch = (shape.size() > 1) ? shape[0] : 1;
    int N = (shape.size() > 1) ? shape[1] : shape[0];

    mx::eval(data);
    auto ptr = data.data<int32_t>();

    for (int b = 0; b < batch; ++b) {
        std::vector<uint32_t> poly(N);
        for (int i = 0; i < N; ++i) {
            poly[i] = static_cast<uint32_t>(ptr[b * N + i]);
        }
        ntt_forward_cpu(poly);
        for (int i = 0; i < N; ++i) {
            const_cast<int32_t*>(ptr)[b * N + i] = static_cast<int32_t>(poly[i]);
        }
    }

    // Re-create array with updated values
    std::vector<int32_t> result(ptr, ptr + batch * N);
    data = mx::array(result.data(), shape, mx::int32);
    mx::eval(data);
}

inline void Barrett32Dispatcher::ntt_inverse(mx::array& data) {
    auto shape = data.shape();
    int batch = (shape.size() > 1) ? shape[0] : 1;
    int N = (shape.size() > 1) ? shape[1] : shape[0];

    mx::eval(data);
    auto ptr = data.data<int32_t>();

    for (int b = 0; b < batch; ++b) {
        std::vector<uint32_t> poly(N);
        for (int i = 0; i < N; ++i) {
            poly[i] = static_cast<uint32_t>(ptr[b * N + i]);
        }
        ntt_inverse_cpu(poly);
        for (int i = 0; i < N; ++i) {
            const_cast<int32_t*>(ptr)[b * N + i] = static_cast<int32_t>(poly[i]);
        }
    }

    std::vector<int32_t> result(ptr, ptr + batch * N);
    data = mx::array(result.data(), shape, mx::int32);
    mx::eval(data);
}

inline mx::array Barrett32Dispatcher::pointwise_mul(const mx::array& a, const mx::array& b) {
    return mul_mod(a, b);
}

inline mx::array Barrett32Dispatcher::poly_mul(const mx::array& a, const mx::array& b) {
    auto a_ntt = mx::array(a);
    auto b_ntt = mx::array(b);

    ntt_forward(a_ntt);
    ntt_forward(b_ntt);

    auto prod = pointwise_mul(a_ntt, b_ntt);

    ntt_inverse(prod);

    return prod;
}

#endif // WITH_MLX

// =============================================================================
// Test Utilities
// =============================================================================

inline bool test_barrett32_correctness() {
    // Test with known NTT-friendly prime
    uint32_t Q = 998244353;  // 7 * 17 * 2^23 + 1
    uint32_t N = 1024;

    Barrett32Params params = Barrett32Params::create(Q, N);

    bool passed = true;

    // Test 1: Barrett multiply-reduce
    {
        uint32_t a = 12345678;
        uint32_t b = 87654321;
        uint32_t expected = static_cast<uint32_t>(
            (static_cast<uint64_t>(a) * b) % Q
        );
        uint32_t result = barrett_mul_mod_cpu(a, b, Q, params.mu);

        if (result != expected) {
            passed = false;
        }
    }

    // Test 2: Boundary cases
    {
        // a * b = Q - 1 (near modulus)
        uint32_t a = Q - 1;
        uint32_t b = 1;
        uint32_t result = barrett_mul_mod_cpu(a, b, Q, params.mu);
        if (result != Q - 1) passed = false;

        // a * b = 0
        result = barrett_mul_mod_cpu(0, 12345, Q, params.mu);
        if (result != 0) passed = false;

        // a * b = 1
        result = barrett_mul_mod_cpu(1, 1, Q, params.mu);
        if (result != 1) passed = false;
    }

    // Test 3: Modular add/sub
    {
        uint32_t a = Q - 100;
        uint32_t b = 200;
        uint32_t sum = mod_add_cpu(a, b, Q);
        if (sum != 100) passed = false;

        uint32_t diff = mod_sub_cpu(100, 200, Q);
        if (diff != Q - 100) passed = false;
    }

    // Test 4: Constant-time property (timing should be same for all inputs)
    // This is a smoke test - real timing analysis needs proper tooling
    {
        uint32_t results[4];
        results[0] = barrett_mul_mod_cpu(0, 0, Q, params.mu);
        results[1] = barrett_mul_mod_cpu(Q-1, Q-1, Q, params.mu);
        results[2] = barrett_mul_mod_cpu(Q/2, Q/2, Q, params.mu);
        results[3] = barrett_mul_mod_cpu(1, 1, Q, params.mu);
        // All should complete (no infinite loops, crashes)
        (void)results;
    }

    return passed;
}

#ifdef WITH_MLX

inline bool test_barrett32_ntt() {
    // Test NTT with small prime
    uint32_t Q = 998244353;
    uint32_t N = 16;  // Small for testing

    Barrett32Dispatcher dispatcher(Q, N);

    if (!dispatcher.is_available()) {
        // Skip GPU tests if Metal not available
        return true;
    }

    bool passed = true;

    // Test 1: NTT(INTT(x)) == x
    {
        std::vector<int32_t> original(N);
        for (uint32_t i = 0; i < N; ++i) {
            original[i] = static_cast<int32_t>(i * 12345 % Q);
        }

        auto data = mx::array(original.data(), {static_cast<int>(N)}, mx::int32);
        mx::eval(data);

        dispatcher.ntt_forward(data);
        dispatcher.ntt_inverse(data);

        mx::eval(data);
        auto ptr = data.data<int32_t>();

        for (uint32_t i = 0; i < N; ++i) {
            if (static_cast<uint32_t>(ptr[i]) != static_cast<uint32_t>(original[i])) {
                passed = false;
                break;
            }
        }
    }

    // Test 2: Polynomial multiplication via NTT
    {
        // Multiply (1 + x) * (1 + x) = 1 + 2x + x^2
        std::vector<int32_t> a(N, 0), b(N, 0);
        a[0] = 1; a[1] = 1;
        b[0] = 1; b[1] = 1;

        auto a_arr = mx::array(a.data(), {static_cast<int>(N)}, mx::int32);
        auto b_arr = mx::array(b.data(), {static_cast<int>(N)}, mx::int32);

        auto result = dispatcher.poly_mul(a_arr, b_arr);
        mx::eval(result);

        auto ptr = result.data<int32_t>();

        // Expected: [1, 2, 1, 0, 0, ...]
        if (ptr[0] != 1 || ptr[1] != 2 || ptr[2] != 1) {
            passed = false;
        }
        for (uint32_t i = 3; i < N; ++i) {
            if (ptr[i] != 0) {
                passed = false;
                break;
            }
        }
    }

    // Test 3: Batch processing
    {
        int batch = 4;
        std::vector<int32_t> data(batch * N);
        for (int b = 0; b < batch; ++b) {
            for (uint32_t i = 0; i < N; ++i) {
                data[b * N + i] = static_cast<int32_t>((b * 1000 + i) % Q);
            }
        }

        auto original = data;
        auto arr = mx::array(data.data(), {batch, static_cast<int>(N)}, mx::int32);

        dispatcher.ntt_forward(arr);
        dispatcher.ntt_inverse(arr);

        mx::eval(arr);
        auto ptr = arr.data<int32_t>();

        for (int i = 0; i < batch * static_cast<int>(N); ++i) {
            if (ptr[i] != original[i]) {
                passed = false;
                break;
            }
        }
    }

    return passed;
}

#endif // WITH_MLX

}  // namespace metal
}  // namespace gpu
}  // namespace lbcrypto

#endif // LBCRYPTO_MATH_HAL_MLX_BARRETT_METAL_H
