// =============================================================================
// NTT GPU Engine - MLX Metal Backend for FHE
// =============================================================================
//
// High-performance Number Theoretic Transform using Apple Metal via MLX.
// Features:
// - Barrett reduction with precomputed constants
// - Bit-reversed twiddle factor storage (OpenFHE compatible)
// - Cooley-Tukey forward, Gentleman-Sande inverse
// - Batch processing for Lux FHE blind rotation
// - GPU acceleration via Metal compute shaders
//
// Twiddle Prefetch Optimization (2025):
// - Stage-indexed twiddle layout for sequential memory access
// - Shared memory prefetch eliminates global memory bottleneck
// - For N <= 4096: all twiddles fit in M3's 32KB shared memory
// - Estimated 10x speedup on twiddle access latency
// - See ntt_twiddle_cache.h for custom Metal kernel implementation
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

#ifndef LUX_FHE_MATH_HAL_MLX_NTT_H
#define LUX_FHE_MATH_HAL_MLX_NTT_H

#include <cstdint>
#include <vector>
#include <memory>
#include <stdexcept>

#ifdef WITH_MLX
#include <mlx/mlx.h>
#include "metal_dispatch.h"
#include "ntt_twiddle_cache.h"
namespace mx = mlx::core;
#endif

namespace lux {
namespace gpu {

// =============================================================================
// Modular Arithmetic Utilities
// =============================================================================

inline void extended_gcd(uint64_t a, uint64_t b,
                         int64_t& g, int64_t& x, int64_t& y) {
    if (b == 0) { g = a; x = 1; y = 0; return; }
    int64_t g1, x1, y1;
    extended_gcd(b, a % b, g1, x1, y1);
    g = g1; x = y1; y = x1 - (int64_t)(a / b) * y1;
}

inline uint64_t mod_inverse(uint64_t a, uint64_t m) {
    int64_t g, x, y;
    extended_gcd(a, m, g, x, y);
    if (g != 1) throw std::runtime_error("Modular inverse does not exist");
    return (x % (int64_t)m + m) % m;
}

inline uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m) {
    return static_cast<uint64_t>((__uint128_t)a * b % m);
}

inline uint64_t powmod(uint64_t base, uint64_t exp, uint64_t m) {
    uint64_t result = 1;
    base %= m;
    while (exp > 0) {
        if (exp & 1) result = mulmod(result, base, m);
        base = mulmod(base, base, m);
        exp >>= 1;
    }
    return result;
}

inline uint32_t bit_reverse(uint32_t x, uint32_t bits) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < bits; ++i) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// Find primitive 2N-th root of unity modulo Q
inline uint64_t find_primitive_root(uint32_t N, uint64_t Q) {
    uint64_t order = Q - 1;
    if (order % (2 * N) != 0) {
        throw std::runtime_error("Q - 1 must be divisible by 2N");
    }
    for (uint64_t g = 2; g < Q; ++g) {
        if (powmod(g, order / 2, Q) != 1) {
            return powmod(g, order / (2 * N), Q);
        }
    }
    throw std::runtime_error("No primitive root found");
}

// =============================================================================
// NTT Parameters (matches Metal shader struct)
// =============================================================================

struct NTTParams {
    uint64_t Q;            // Prime modulus
    uint64_t mu;           // Barrett constant: floor(2^64 / Q)
    uint64_t N_inv;        // N^{-1} mod Q
    uint64_t N_inv_precon; // Barrett precomputation for N_inv
    uint32_t N;            // Ring dimension
    uint32_t log_N;        // log2(N)

    static NTTParams create(uint32_t N, uint64_t Q) {
        NTTParams p;
        p.N = N;
        p.Q = Q;
        p.log_N = 0;
        while ((1u << p.log_N) < N) ++p.log_N;
        if ((1u << p.log_N) != N) throw std::runtime_error("N must be power of 2");

        // Barrett constant: floor(2^64 / Q)
        p.mu = static_cast<uint64_t>((__uint128_t)1 << 64) / Q;
        p.N_inv = mod_inverse(N, Q);
        p.N_inv_precon = static_cast<uint64_t>(((__uint128_t)p.N_inv << 64) / Q);
        return p;
    }
};

// =============================================================================
// Twiddle Factor Computation (OpenFHE bit-reversed storage)
// =============================================================================

inline void compute_twiddles(uint32_t N, uint64_t Q,
                             std::vector<uint64_t>& tw,
                             std::vector<uint64_t>& tw_precon) {
    uint64_t omega = find_primitive_root(N, Q);
    tw.resize(N);
    tw_precon.resize(N);

    uint32_t log_N = 0;
    while ((1u << log_N) < N) ++log_N;

    for (uint32_t m = 1; m < N; m <<= 1) {
        uint32_t log_m = 0;
        while ((1u << log_m) < m) ++log_m;
        for (uint32_t i = 0; i < m; ++i) {
            uint32_t exp = (N / m) * bit_reverse(i, log_m);
            tw[m + i] = powmod(omega, exp, Q);
            tw_precon[m + i] = static_cast<uint64_t>(((__uint128_t)tw[m + i] << 64) / Q);
        }
    }
    tw[0] = 1;
    tw_precon[0] = static_cast<uint64_t>(((__uint128_t)1 << 64) / Q);
}

inline void compute_inv_twiddles(uint32_t N, uint64_t Q,
                                  std::vector<uint64_t>& tw,
                                  std::vector<uint64_t>& tw_precon) {
    uint64_t omega_inv = mod_inverse(find_primitive_root(N, Q), Q);
    tw.resize(N);
    tw_precon.resize(N);

    for (uint32_t m = 1; m < N; m <<= 1) {
        uint32_t log_m = 0;
        while ((1u << log_m) < m) ++log_m;
        for (uint32_t i = 0; i < m; ++i) {
            uint32_t exp = (N / m) * bit_reverse(i, log_m);
            tw[m + i] = powmod(omega_inv, exp, Q);
            tw_precon[m + i] = static_cast<uint64_t>(((__uint128_t)tw[m + i] << 64) / Q);
        }
    }
    tw[0] = 1;
    tw_precon[0] = static_cast<uint64_t>(((__uint128_t)1 << 64) / Q);
}

// =============================================================================
// CPU Modular Arithmetic (Reference/Fallback)
// =============================================================================

inline uint64_t barrett_mul(uint64_t a, uint64_t w, uint64_t Q, uint64_t precon) {
    uint64_t q_approx = static_cast<uint64_t>(((__uint128_t)a * precon) >> 64);
    uint64_t result = a * w - q_approx * Q;
    return (result >= Q) ? result - Q : result;
}

inline uint64_t mod_add(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t sum = a + b;
    return (sum >= Q) ? sum - Q : sum;
}

inline uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t Q) {
    return (a >= b) ? a - b : a + Q - b;
}

// =============================================================================
// NTT Engine
// =============================================================================

#ifdef WITH_MLX

class NTTEngine {
public:
    explicit NTTEngine(uint32_t N, uint64_t Q);

    // NTT transforms
    void forward(mx::array& data);
    void inverse(mx::array& data);

    // Polynomial operations
    mx::array pointwise_mul(const mx::array& a, const mx::array& b);
    mx::array poly_mul(const mx::array& a, const mx::array& b);
    mx::array rotate(const mx::array& data, const std::vector<int32_t>& rotations);

    const NTTParams& params() const { return params_; }
    bool is_gpu_enabled() const { return use_gpu_; }

private:
    NTTParams params_;
    std::vector<uint64_t> tw_, tw_precon_;
    std::vector<uint64_t> inv_tw_, inv_tw_precon_;
    std::shared_ptr<mx::array> tw_gpu_, tw_precon_gpu_;
    std::shared_ptr<mx::array> inv_tw_gpu_, inv_tw_precon_gpu_;
    bool gpu_ready_ = false;
    bool use_gpu_ = false;

    // Metal dispatcher for GPU acceleration
    std::unique_ptr<metal::NTTMetalDispatcher> metal_dispatcher_;

    void init_gpu();
    void forward_cpu(std::vector<uint64_t>& data);
    void inverse_cpu(std::vector<uint64_t>& data);
};

// Implementation

inline NTTEngine::NTTEngine(uint32_t N, uint64_t Q) : params_(NTTParams::create(N, Q)) {
    compute_twiddles(N, Q, tw_, tw_precon_);
    compute_inv_twiddles(N, Q, inv_tw_, inv_tw_precon_);

    // Try to initialize Metal GPU dispatcher
    try {
        if (mx::metal::is_available()) {
            metal_dispatcher_ = std::make_unique<metal::NTTMetalDispatcher>(N, Q);
            use_gpu_ = metal_dispatcher_->is_gpu_available();
        }
    } catch (...) {
        use_gpu_ = false;
    }
}

inline void NTTEngine::init_gpu() {
    if (gpu_ready_) return;

    int N = static_cast<int>(params_.N);
    std::vector<int64_t> tw_i64(tw_.begin(), tw_.end());
    std::vector<int64_t> tw_precon_i64(tw_precon_.begin(), tw_precon_.end());
    std::vector<int64_t> inv_tw_i64(inv_tw_.begin(), inv_tw_.end());
    std::vector<int64_t> inv_tw_precon_i64(inv_tw_precon_.begin(), inv_tw_precon_.end());

    tw_gpu_ = std::make_shared<mx::array>(mx::array(tw_i64.data(), {N}, mx::int64));
    tw_precon_gpu_ = std::make_shared<mx::array>(mx::array(tw_precon_i64.data(), {N}, mx::int64));
    inv_tw_gpu_ = std::make_shared<mx::array>(mx::array(inv_tw_i64.data(), {N}, mx::int64));
    inv_tw_precon_gpu_ = std::make_shared<mx::array>(mx::array(inv_tw_precon_i64.data(), {N}, mx::int64));

    mx::eval(*tw_gpu_);
    mx::eval(*tw_precon_gpu_);
    mx::eval(*inv_tw_gpu_);
    mx::eval(*inv_tw_precon_gpu_);
    gpu_ready_ = true;
}

inline void NTTEngine::forward_cpu(std::vector<uint64_t>& data) {
    uint32_t N = params_.N;
    uint64_t Q = params_.Q;

    for (uint32_t s = 0; s < params_.log_N; ++s) {
        uint32_t m = 1u << s;
        uint32_t t = N >> (s + 1);

        for (uint32_t i = 0; i < m; ++i) {
            uint32_t j1 = i << (params_.log_N - s);
            uint32_t j2 = j1 + t;
            uint64_t w = tw_[m + i];
            uint64_t precon = tw_precon_[m + i];

            for (uint32_t j = j1; j < j2; ++j) {
                uint64_t lo = data[j];
                uint64_t hi = data[j + t];
                uint64_t whi = barrett_mul(hi, w, Q, precon);
                data[j] = mod_add(lo, whi, Q);
                data[j + t] = mod_sub(lo, whi, Q);
            }
        }
    }
}

inline void NTTEngine::inverse_cpu(std::vector<uint64_t>& data) {
    uint32_t N = params_.N;
    uint64_t Q = params_.Q;

    for (uint32_t s = 0; s < params_.log_N; ++s) {
        uint32_t m = N >> (s + 1);
        uint32_t t = 1u << s;

        for (uint32_t i = 0; i < m; ++i) {
            uint32_t j1 = i << (s + 1);
            uint32_t j2 = j1 + t;
            uint64_t w = inv_tw_[m + i];
            uint64_t precon = inv_tw_precon_[m + i];

            for (uint32_t j = j1; j < j2; ++j) {
                uint64_t lo = data[j];
                uint64_t hi = data[j + t];
                data[j] = mod_add(lo, hi, Q);
                data[j + t] = barrett_mul(mod_sub(lo, hi, Q), w, Q, precon);
            }
        }
    }

    // Scale by N^{-1}
    for (uint32_t i = 0; i < N; ++i) {
        data[i] = barrett_mul(data[i], params_.N_inv, Q, params_.N_inv_precon);
    }
}

inline void NTTEngine::forward(mx::array& data) {
    // Try GPU path first
    if (use_gpu_ && metal_dispatcher_) {
        try {
            metal_dispatcher_->forward(data);
            return;
        } catch (...) {
            // Fall through to CPU
        }
    }

    // CPU fallback
    auto shape = data.shape();
    int batch = (shape.size() > 1) ? shape[0] : 1;
    int N = (shape.size() > 1) ? shape[1] : shape[0];

    mx::eval(data);
    auto ptr = data.data<int64_t>();
    std::vector<uint64_t> work(ptr, ptr + batch * N);

    for (int b = 0; b < batch; ++b) {
        std::vector<uint64_t> poly(work.begin() + b * N, work.begin() + (b + 1) * N);
        forward_cpu(poly);
        std::copy(poly.begin(), poly.end(), work.begin() + b * N);
    }

    std::vector<int64_t> result(work.begin(), work.end());
    data = mx::array(result.data(), shape, mx::int64);
    mx::eval(data);
}

inline void NTTEngine::inverse(mx::array& data) {
    // Try GPU path first
    if (use_gpu_ && metal_dispatcher_) {
        try {
            metal_dispatcher_->inverse(data);
            return;
        } catch (...) {
            // Fall through to CPU
        }
    }

    // CPU fallback
    auto shape = data.shape();
    int batch = (shape.size() > 1) ? shape[0] : 1;
    int N = (shape.size() > 1) ? shape[1] : shape[0];

    mx::eval(data);
    auto ptr = data.data<int64_t>();
    std::vector<uint64_t> work(ptr, ptr + batch * N);

    for (int b = 0; b < batch; ++b) {
        std::vector<uint64_t> poly(work.begin() + b * N, work.begin() + (b + 1) * N);
        inverse_cpu(poly);
        std::copy(poly.begin(), poly.end(), work.begin() + b * N);
    }

    std::vector<int64_t> result(work.begin(), work.end());
    data = mx::array(result.data(), shape, mx::int64);
    mx::eval(data);
}

inline mx::array NTTEngine::pointwise_mul(const mx::array& a, const mx::array& b) {
    // Try GPU path first
    if (use_gpu_ && metal_dispatcher_) {
        try {
            return metal_dispatcher_->pointwise_mul(a, b);
        } catch (...) {
            // Fall through to CPU
        }
    }

    // CPU fallback for correct modular multiplication
    auto shape = a.shape();
    int batch = (shape.size() > 1) ? shape[0] : 1;
    int N = (shape.size() > 1) ? shape[1] : shape[0];
    uint64_t Q = params_.Q;

    mx::eval(a);
    mx::eval(b);
    auto a_ptr = a.data<int64_t>();
    auto b_ptr = b.data<int64_t>();
    std::vector<int64_t> result(batch * N);

    for (int i = 0; i < batch * N; ++i) {
        uint64_t av = static_cast<uint64_t>(a_ptr[i]);
        uint64_t bv = static_cast<uint64_t>(b_ptr[i]);
        result[i] = static_cast<int64_t>(mulmod(av, bv, Q));
    }

    return mx::array(result.data(), shape, mx::int64);
}

inline mx::array NTTEngine::poly_mul(const mx::array& a, const mx::array& b) {
    auto a_ntt = mx::array(a);
    auto b_ntt = mx::array(b);
    forward(a_ntt);
    forward(b_ntt);
    auto prod = pointwise_mul(a_ntt, b_ntt);
    inverse(prod);
    return prod;
}

inline mx::array NTTEngine::rotate(const mx::array& data,
                                    const std::vector<int32_t>& rotations) {
    auto shape = data.shape();
    int batch = shape[0];
    int N = shape[1];
    uint64_t Q = params_.Q;

    mx::eval(data);
    auto ptr = data.data<int64_t>();
    std::vector<int64_t> result(batch * N);

    for (int b = 0; b < batch; ++b) {
        int32_t k = rotations[b];
        int32_t two_N = 2 * N;
        k = ((k % two_N) + two_N) % two_N;

        for (int i = 0; i < N; ++i) {
            int32_t src = i - k;
            bool neg = false;
            while (src < 0) { src += N; neg = !neg; }
            while (src >= N) { src -= N; neg = !neg; }

            uint64_t val = static_cast<uint64_t>(ptr[b * N + src]);
            result[b * N + i] = neg ? static_cast<int64_t>(Q - val)
                                    : static_cast<int64_t>(val);
        }
    }

    return mx::array(result.data(), shape, mx::int64);
}

#endif // WITH_MLX

}  // namespace gpu
}  // namespace lux::fhe

#endif // LUX_FHE_MATH_HAL_MLX_NTT_H
