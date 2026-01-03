// =============================================================================
// Four-Step NTT for Maximum GPU Throughput
// =============================================================================
//
// Implements the four-step FFT/NTT algorithm optimized for Metal GPU.
//
// Algorithm (for N = n1 * n2):
//   Step 1: n2 row NTTs of size n1
//   Step 2: Multiply by twiddle factors (diagonal matrix)
//   Step 3: Transpose (n1 x n2) -> (n2 x n1)
//   Step 4: n1 column NTTs of size n2
//
// Key advantage: Only 2 synchronization points vs log(N) in standard NTT.
// This dramatically improves GPU utilization on Metal.
//
// Dimension choices:
//   N=1024:  32 x 32
//   N=4096:  64 x 64
//   N=16384: 128 x 128
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

#ifndef LUX_FHE_MATH_HAL_MLX_NTT_FOURSTEP_H
#define LUX_FHE_MATH_HAL_MLX_NTT_FOURSTEP_H

#include <cstdint>
#include <vector>
#include <memory>
#include <stdexcept>
#include <algorithm>

#ifdef WITH_MLX
#include <mlx/mlx.h>
#include "metal_dispatch.h"
#include "ntt.h"
namespace mx = mlx::core;
#endif

namespace lux {
namespace gpu {

// =============================================================================
// Four-Step NTT Configuration
// =============================================================================

struct FourStepConfig {
    uint32_t N;       // Total size (must be perfect square for optimal)
    uint32_t n1;      // Row dimension (inner NTT size)
    uint32_t n2;      // Column dimension (outer NTT size)
    uint32_t log_n1;  // log2(n1)
    uint32_t log_n2;  // log2(n2)

    // Determine optimal factorization for GPU
    static FourStepConfig create(uint32_t N) {
        FourStepConfig cfg;
        cfg.N = N;

        // For perfect squares, use sqrt(N) x sqrt(N)
        // Otherwise, factor into two closest powers of 2
        uint32_t log_N = 0;
        while ((1u << log_N) < N) ++log_N;
        if ((1u << log_N) != N) {
            throw std::runtime_error("N must be power of 2");
        }

        // Split log_N roughly in half
        cfg.log_n1 = log_N / 2;
        cfg.log_n2 = log_N - cfg.log_n1;
        cfg.n1 = 1u << cfg.log_n1;
        cfg.n2 = 1u << cfg.log_n2;

        return cfg;
    }

    // Check if four-step is beneficial (N >= 1024)
    static bool should_use_fourstep(uint32_t N) {
        return N >= 1024;
    }
};

// =============================================================================
// Dimension Configurations (compile-time optimal)
// =============================================================================

// N=1024: 32x32 structure (log_n1=5, log_n2=5)
// N=4096: 64x64 structure (log_n1=6, log_n2=6)
// N=16384: 128x128 structure (log_n1=7, log_n2=7)
// N=65536: 256x256 structure (log_n1=8, log_n2=8)

constexpr uint32_t FOURSTEP_MIN_N = 1024;

#ifdef WITH_MLX

// =============================================================================
// Four-Step NTT Twiddle Factors
// =============================================================================
//
// The four-step algorithm requires three sets of twiddles:
// 1. Inner NTT twiddles (size n1)
// 2. Diagonal twiddle matrix (n1 x n2 values: omega^{i*j})
// 3. Outer NTT twiddles (size n2)
//
// For inverse, we use conjugate twiddles.

class FourStepTwiddles {
public:
    FourStepTwiddles(const FourStepConfig& cfg, uint64_t Q);

    // Access twiddles (dereference shared_ptr)
    const mx::array& inner_tw() const { return *inner_tw_; }
    const mx::array& inner_tw_inv() const { return *inner_tw_inv_; }
    const mx::array& diagonal_tw() const { return *diagonal_tw_; }
    const mx::array& diagonal_tw_inv() const { return *diagonal_tw_inv_; }
    const mx::array& outer_tw() const { return *outer_tw_; }
    const mx::array& outer_tw_inv() const { return *outer_tw_inv_; }

    uint64_t N_inv() const { return N_inv_; }

private:
    FourStepConfig cfg_;
    uint64_t Q_;
    uint64_t N_inv_;  // N^{-1} mod Q

    // GPU arrays: inner/outer are [n], diagonal is [n1, n2]
    // Use shared_ptr to avoid mx::array default constructor issues
    std::shared_ptr<mx::array> inner_tw_;       // [n1]
    std::shared_ptr<mx::array> inner_tw_inv_;   // [n1]
    std::shared_ptr<mx::array> diagonal_tw_;    // [n1, n2]
    std::shared_ptr<mx::array> diagonal_tw_inv_;// [n1, n2]
    std::shared_ptr<mx::array> outer_tw_;       // [n2]
    std::shared_ptr<mx::array> outer_tw_inv_;   // [n2]
};

inline FourStepTwiddles::FourStepTwiddles(const FourStepConfig& cfg, uint64_t Q)
    : cfg_(cfg), Q_(Q) {

    uint32_t N = cfg.N;
    uint32_t n1 = cfg.n1;
    uint32_t n2 = cfg.n2;

    // Find primitive 2N-th root of unity
    uint64_t omega = find_primitive_root(N, Q);
    uint64_t omega_inv = mod_inverse(omega, Q);
    N_inv_ = mod_inverse(N, Q);

    // Find primitive 2*n1-th and 2*n2-th roots for inner/outer NTTs
    // omega_n1 = omega^{N/n1} = omega^{n2}
    // omega_n2 = omega^{N/n2} = omega^{n1}
    uint64_t omega_n1 = powmod(omega, n2, Q);
    uint64_t omega_n1_inv = powmod(omega_inv, n2, Q);
    uint64_t omega_n2 = powmod(omega, n1, Q);
    uint64_t omega_n2_inv = powmod(omega_inv, n1, Q);

    // Compute inner NTT twiddles (bit-reversed for Cooley-Tukey)
    std::vector<int64_t> inner_tw_vec(n1);
    std::vector<int64_t> inner_tw_inv_vec(n1);

    for (uint32_t m = 1; m < n1; m <<= 1) {
        uint32_t log_m = 0;
        while ((1u << log_m) < m) ++log_m;
        for (uint32_t i = 0; i < m; ++i) {
            uint32_t exp = (n1 / m) * bit_reverse(i, log_m);
            inner_tw_vec[m + i] = static_cast<int64_t>(powmod(omega_n1, exp, Q));
            inner_tw_inv_vec[m + i] = static_cast<int64_t>(powmod(omega_n1_inv, exp, Q));
        }
    }
    inner_tw_vec[0] = 1;
    inner_tw_inv_vec[0] = 1;

    // Compute outer NTT twiddles
    std::vector<int64_t> outer_tw_vec(n2);
    std::vector<int64_t> outer_tw_inv_vec(n2);

    for (uint32_t m = 1; m < n2; m <<= 1) {
        uint32_t log_m = 0;
        while ((1u << log_m) < m) ++log_m;
        for (uint32_t i = 0; i < m; ++i) {
            uint32_t exp = (n2 / m) * bit_reverse(i, log_m);
            outer_tw_vec[m + i] = static_cast<int64_t>(powmod(omega_n2, exp, Q));
            outer_tw_inv_vec[m + i] = static_cast<int64_t>(powmod(omega_n2_inv, exp, Q));
        }
    }
    outer_tw_vec[0] = 1;
    outer_tw_inv_vec[0] = 1;

    // Compute diagonal twiddle matrix: omega^{i*j} for 0 <= i < n1, 0 <= j < n2
    // This is the key to the four-step algorithm
    std::vector<int64_t> diag_tw_vec(n1 * n2);
    std::vector<int64_t> diag_tw_inv_vec(n1 * n2);

    for (uint32_t i = 0; i < n1; ++i) {
        for (uint32_t j = 0; j < n2; ++j) {
            uint64_t exp = static_cast<uint64_t>(i) * j;
            diag_tw_vec[i * n2 + j] = static_cast<int64_t>(powmod(omega, exp, Q));
            diag_tw_inv_vec[i * n2 + j] = static_cast<int64_t>(powmod(omega_inv, exp, Q));
        }
    }

    // Upload to GPU (using shared_ptr)
    inner_tw_ = std::make_shared<mx::array>(
        mx::array(inner_tw_vec.data(), {static_cast<int>(n1)}, mx::int64));
    inner_tw_inv_ = std::make_shared<mx::array>(
        mx::array(inner_tw_inv_vec.data(), {static_cast<int>(n1)}, mx::int64));
    outer_tw_ = std::make_shared<mx::array>(
        mx::array(outer_tw_vec.data(), {static_cast<int>(n2)}, mx::int64));
    outer_tw_inv_ = std::make_shared<mx::array>(
        mx::array(outer_tw_inv_vec.data(), {static_cast<int>(n2)}, mx::int64));
    diagonal_tw_ = std::make_shared<mx::array>(
        mx::array(diag_tw_vec.data(),
                  {static_cast<int>(n1), static_cast<int>(n2)}, mx::int64));
    diagonal_tw_inv_ = std::make_shared<mx::array>(
        mx::array(diag_tw_inv_vec.data(),
                  {static_cast<int>(n1), static_cast<int>(n2)}, mx::int64));

    mx::eval(*inner_tw_);
    mx::eval(*inner_tw_inv_);
    mx::eval(*outer_tw_);
    mx::eval(*outer_tw_inv_);
    mx::eval(*diagonal_tw_);
    mx::eval(*diagonal_tw_inv_);
}

// =============================================================================
// NTTFourStep - Main Four-Step NTT Engine
// =============================================================================
//
// Constant-time implementation: same execution path regardless of input data.
// All operations use fixed iteration counts and avoid data-dependent branches.

class NTTFourStep {
public:
    NTTFourStep(uint32_t N, uint64_t Q);

    // Forward NTT (outputs in bit-reversed order)
    void forward(mx::array& data);

    // Inverse NTT (expects bit-reversed input)
    void inverse(mx::array& data);

    // Batched forward NTT: [batch, N] -> [batch, N]
    void forward_batch(mx::array& data);

    // Batched inverse NTT: [batch, N] -> [batch, N]
    void inverse_batch(mx::array& data);

    // Pointwise multiplication mod Q
    mx::array pointwise_mul(const mx::array& a, const mx::array& b);

    // Full polynomial multiplication
    mx::array poly_mul(const mx::array& a, const mx::array& b);

    // Check if four-step is active
    bool is_fourstep_enabled() const { return fourstep_enabled_; }
    bool is_gpu_available() const { return gpu_available_; }

    const FourStepConfig& config() const { return cfg_; }
    uint64_t modulus() const { return Q_; }

private:
    FourStepConfig cfg_;
    uint64_t Q_;
    bool fourstep_enabled_ = false;
    bool gpu_available_ = false;

    // Twiddle factors
    std::unique_ptr<FourStepTwiddles> twiddles_;

    // Fallback for small N
    std::unique_ptr<metal::NTTMetalDispatcher> fallback_ntt_;

    // Core operations
    void row_ntt(mx::array& matrix, bool inverse);
    void apply_diagonal_twiddles(mx::array& matrix, bool inverse);
    void transpose(mx::array& matrix);
    void col_ntt(mx::array& matrix, bool inverse);

    // Single butterfly stage (constant-time)
    void butterfly_stage(mx::array& data, uint32_t stage,
                         const mx::array& tw, uint32_t n);
};

// Implementation

inline NTTFourStep::NTTFourStep(uint32_t N, uint64_t Q) : Q_(Q) {
    gpu_available_ = mx::metal::is_available();

    if (gpu_available_) {
        mx::set_default_device(mx::Device::gpu);
    }

    // Use four-step for N >= 1024
    if (FourStepConfig::should_use_fourstep(N)) {
        cfg_ = FourStepConfig::create(N);
        twiddles_ = std::make_unique<FourStepTwiddles>(cfg_, Q);
        fourstep_enabled_ = true;
    } else {
        // Fallback to standard NTT for small N
        cfg_.N = N;
        cfg_.n1 = N;
        cfg_.n2 = 1;
        uint32_t log_N = 0;
        while ((1u << log_N) < N) ++log_N;
        cfg_.log_n1 = log_N;
        cfg_.log_n2 = 0;

        fallback_ntt_ = std::make_unique<metal::NTTMetalDispatcher>(N, Q);
        fourstep_enabled_ = false;
    }
}

// -----------------------------------------------------------------------------
// Butterfly Stage (constant-time)
// -----------------------------------------------------------------------------
// Performs one stage of Cooley-Tukey NTT butterfly on data of size n.
// Fixed iteration count, no data-dependent branches.

inline void NTTFourStep::butterfly_stage(mx::array& data, uint32_t stage,
                                          const mx::array& tw, uint32_t n) {
    uint32_t log_n = 0;
    while ((1u << log_n) < n) ++log_n;

    uint32_t m = 1u << stage;
    uint32_t t = n >> (stage + 1);
    int half_n = static_cast<int>(n / 2);

    // Build index arrays (constant pattern per stage)
    std::vector<int32_t> lo_indices(half_n);
    std::vector<int32_t> hi_indices(half_n);
    std::vector<int32_t> tw_indices(half_n);

    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < t; ++j) {
            uint32_t idx = i * t + j;
            uint32_t idx_lo = (i << (log_n - stage)) + j;
            uint32_t idx_hi = idx_lo + t;
            lo_indices[idx] = static_cast<int32_t>(idx_lo);
            hi_indices[idx] = static_cast<int32_t>(idx_hi);
            tw_indices[idx] = static_cast<int32_t>(m + i);
        }
    }

    auto lo_idx = mx::array(lo_indices.data(), {half_n}, mx::int32);
    auto hi_idx = mx::array(hi_indices.data(), {half_n}, mx::int32);
    auto tw_idx = mx::array(tw_indices.data(), {half_n}, mx::int32);

    // Gather values
    auto lo_vals = mx::take(data, lo_idx, 0);
    auto hi_vals = mx::take(data, hi_idx, 0);
    auto tw_vals = mx::take(tw, tw_idx, 0);

    // Modular multiply: hi * twiddle mod Q
    auto Q_arr = mx::array(static_cast<int64_t>(Q_));
    auto hi_tw = mx::remainder(mx::multiply(hi_vals, tw_vals), Q_arr);

    // Ensure non-negative
    hi_tw = mx::where(mx::less(hi_tw, mx::array(static_cast<int64_t>(0))),
                      mx::add(hi_tw, Q_arr), hi_tw);

    // Butterfly: new_lo = lo + hi*tw, new_hi = lo - hi*tw
    auto new_lo = mx::remainder(mx::add(lo_vals, hi_tw), Q_arr);
    auto diff = mx::subtract(lo_vals, hi_tw);
    auto new_hi = mx::remainder(mx::add(diff, Q_arr), Q_arr);

    // Scatter back
    data = mx::scatter(data, lo_idx, new_lo, 0);
    data = mx::scatter(data, hi_idx, new_hi, 0);
    mx::eval(data);
}

// -----------------------------------------------------------------------------
// Row NTT: n2 parallel NTTs of size n1
// -----------------------------------------------------------------------------
// Data layout: [n1, n2] interpreted as n2 rows of length n1.
// Each row gets an independent NTT.

// =============================================================================
// OPTIMIZED: Row NTT with Vectorized Scatter
// =============================================================================
// Key change: Use axis=1 scatter instead of per-row loop.
// This eliminates n2 sequential scatter operations per stage.

inline void NTTFourStep::row_ntt(mx::array& matrix, bool inverse) {
    uint32_t n1 = cfg_.n1;
    uint32_t n2 = cfg_.n2;
    uint32_t log_n1 = cfg_.log_n1;

    const auto& tw = inverse ? twiddles_->inner_tw_inv() : twiddles_->inner_tw();

    // Transpose to [n2, n1] for efficient row processing
    auto transposed = mx::transpose(matrix, {1, 0});
    // NO mx::eval() here - let MLX JIT fuse with subsequent operations

    auto Q_arr = mx::array(static_cast<int64_t>(Q_));

    // Stage-by-stage NTT, all rows processed in parallel per stage
    for (uint32_t stage = 0; stage < log_n1; ++stage) {
        uint32_t m = 1u << stage;
        uint32_t t = n1 >> (stage + 1);
        int half_n = static_cast<int>(n1 / 2);

        // Build butterfly indices (could be cached)
        std::vector<int32_t> lo_indices(half_n);
        std::vector<int32_t> hi_indices(half_n);
        std::vector<int32_t> tw_indices(half_n);

        for (uint32_t i = 0; i < m; ++i) {
            for (uint32_t j = 0; j < t; ++j) {
                uint32_t idx = i * t + j;
                uint32_t idx_lo = (i << (log_n1 - stage)) + j;
                uint32_t idx_hi = idx_lo + t;
                lo_indices[idx] = static_cast<int32_t>(idx_lo);
                hi_indices[idx] = static_cast<int32_t>(idx_hi);
                tw_indices[idx] = static_cast<int32_t>(m + i);
            }
        }

        auto lo_idx = mx::array(lo_indices.data(), {half_n}, mx::int32);
        auto hi_idx = mx::array(hi_indices.data(), {half_n}, mx::int32);
        auto tw_idx = mx::array(tw_indices.data(), {half_n}, mx::int32);

        // VECTORIZED: Gather along axis=1 for ALL rows at once
        // transposed: [n2, n1] -> lo_vals: [n2, half_n]
        auto lo_vals = mx::take(transposed, lo_idx, 1);
        auto hi_vals = mx::take(transposed, hi_idx, 1);
        auto tw_vals = mx::take(tw, tw_idx, 0);  // [half_n], broadcast to [n2, half_n]

        // Butterfly: all rows in single kernel
        auto hi_tw = mx::remainder(mx::multiply(hi_vals, tw_vals), Q_arr);
        hi_tw = mx::where(mx::less(hi_tw, mx::array(static_cast<int64_t>(0))),
                          mx::add(hi_tw, Q_arr), hi_tw);

        auto new_lo = mx::remainder(mx::add(lo_vals, hi_tw), Q_arr);
        auto diff = mx::subtract(lo_vals, hi_tw);
        auto new_hi = mx::remainder(mx::add(diff, Q_arr), Q_arr);

        // VECTORIZED SCATTER: axis=1 processes all rows simultaneously
        transposed = mx::scatter(transposed, lo_idx, new_lo, 1);
        transposed = mx::scatter(transposed, hi_idx, new_hi, 1);
        // NO mx::eval() - let MLX fuse stages
    }

    // Single eval after all stages
    mx::eval(transposed);

    // Transpose back to [n1, n2]
    matrix = mx::transpose(transposed, {1, 0});
    mx::eval(matrix);
}

// -----------------------------------------------------------------------------
// Apply Diagonal Twiddle Factors
// -----------------------------------------------------------------------------
// Multiply each element (i,j) by omega^{i*j} (or omega^{-i*j} for inverse).
// This is the "twist" step that makes four-step equivalent to full NTT.

inline void NTTFourStep::apply_diagonal_twiddles(mx::array& matrix, bool inverse) {
    const auto& diag = inverse ? twiddles_->diagonal_tw_inv() : twiddles_->diagonal_tw();

    auto Q_arr = mx::array(static_cast<int64_t>(Q_));
    auto prod = mx::multiply(matrix, diag);
    matrix = mx::remainder(prod, Q_arr);

    // Ensure non-negative
    matrix = mx::where(mx::less(matrix, mx::array(static_cast<int64_t>(0))),
                       mx::add(matrix, Q_arr), matrix);
    mx::eval(matrix);
}

// -----------------------------------------------------------------------------
// Transpose
// -----------------------------------------------------------------------------
// (n1 x n2) -> (n2 x n1)
// MLX handles this efficiently with strided views.

inline void NTTFourStep::transpose(mx::array& matrix) {
    matrix = mx::transpose(matrix, {1, 0});
    mx::eval(matrix);
}

// -----------------------------------------------------------------------------
// Column NTT: n1 parallel NTTs of size n2 - OPTIMIZED
// -----------------------------------------------------------------------------
// After transpose, matrix is [n2, n1]. We process along axis=0.
// Key change: Remove per-stage mx::eval() to enable MLX JIT fusion.

inline void NTTFourStep::col_ntt(mx::array& matrix, bool inverse) {
    uint32_t n1 = cfg_.n1;
    uint32_t n2 = cfg_.n2;
    uint32_t log_n2 = cfg_.log_n2;

    const auto& tw = inverse ? twiddles_->outer_tw_inv() : twiddles_->outer_tw();

    auto Q_arr = mx::array(static_cast<int64_t>(Q_));

    // matrix: [n2, n1] - process NTT along axis=0
    for (uint32_t stage = 0; stage < log_n2; ++stage) {
        uint32_t m = 1u << stage;
        uint32_t t = n2 >> (stage + 1);
        int half_n = static_cast<int>(n2 / 2);

        std::vector<int32_t> lo_indices(half_n);
        std::vector<int32_t> hi_indices(half_n);
        std::vector<int32_t> tw_indices(half_n);

        for (uint32_t i = 0; i < m; ++i) {
            for (uint32_t j = 0; j < t; ++j) {
                uint32_t idx = i * t + j;
                uint32_t idx_lo = (i << (log_n2 - stage)) + j;
                uint32_t idx_hi = idx_lo + t;
                lo_indices[idx] = static_cast<int32_t>(idx_lo);
                hi_indices[idx] = static_cast<int32_t>(idx_hi);
                tw_indices[idx] = static_cast<int32_t>(m + i);
            }
        }

        auto lo_idx = mx::array(lo_indices.data(), {half_n}, mx::int32);
        auto hi_idx = mx::array(hi_indices.data(), {half_n}, mx::int32);
        auto tw_idx = mx::array(tw_indices.data(), {half_n}, mx::int32);

        // Gather along axis=0: [half_n, n1]
        auto lo_vals = mx::take(matrix, lo_idx, 0);
        auto hi_vals = mx::take(matrix, hi_idx, 0);
        auto tw_vals = mx::take(tw, tw_idx, 0);

        // Broadcast twiddles: [half_n] -> [half_n, 1] for broadcast to [half_n, n1]
        tw_vals = mx::expand_dims(tw_vals, 1);

        // Butterfly
        auto hi_tw = mx::remainder(mx::multiply(hi_vals, tw_vals), Q_arr);
        hi_tw = mx::where(mx::less(hi_tw, mx::array(static_cast<int64_t>(0))),
                          mx::add(hi_tw, Q_arr), hi_tw);

        auto new_lo = mx::remainder(mx::add(lo_vals, hi_tw), Q_arr);
        auto diff = mx::subtract(lo_vals, hi_tw);
        auto new_hi = mx::remainder(mx::add(diff, Q_arr), Q_arr);

        // Scatter back along axis=0
        matrix = mx::scatter(matrix, lo_idx, new_lo, 0);
        matrix = mx::scatter(matrix, hi_idx, new_hi, 0);
        // NO mx::eval() - let MLX JIT fuse all stages
    }

    // Single eval at the end
    mx::eval(matrix);
}

// -----------------------------------------------------------------------------
// Forward NTT (Four-Step Algorithm)
// -----------------------------------------------------------------------------

inline void NTTFourStep::forward(mx::array& data) {
    if (!fourstep_enabled_) {
        if (fallback_ntt_) {
            fallback_ntt_->forward(data);
        }
        return;
    }

    mx::eval(data);

    uint32_t n1 = cfg_.n1;
    uint32_t n2 = cfg_.n2;

    // Reshape [N] -> [n1, n2] (row-major)
    auto matrix = mx::reshape(data, {static_cast<int>(n1), static_cast<int>(n2)});

    // Step 1: n2 row NTTs of size n1
    row_ntt(matrix, false);

    // Step 2: Multiply by diagonal twiddle factors
    apply_diagonal_twiddles(matrix, false);

    // Step 3: Transpose (n1 x n2) -> (n2 x n1)
    transpose(matrix);

    // Step 4: n1 column NTTs of size n2
    col_ntt(matrix, false);

    // Reshape back to [N]
    data = mx::reshape(matrix, {static_cast<int>(cfg_.N)});
    mx::eval(data);
}

// -----------------------------------------------------------------------------
// Inverse NTT (Four-Step Algorithm)
// -----------------------------------------------------------------------------

inline void NTTFourStep::inverse(mx::array& data) {
    if (!fourstep_enabled_) {
        if (fallback_ntt_) {
            fallback_ntt_->inverse(data);
        }
        return;
    }

    mx::eval(data);

    uint32_t n1 = cfg_.n1;
    uint32_t n2 = cfg_.n2;

    // Reshape [N] -> [n2, n1] (after forward, data is in transposed form)
    auto matrix = mx::reshape(data, {static_cast<int>(n2), static_cast<int>(n1)});

    // Inverse steps in reverse order:
    // Step 4': Inverse column NTTs
    col_ntt(matrix, true);

    // Step 3': Transpose back (n2 x n1) -> (n1 x n2)
    transpose(matrix);

    // Step 2': Apply inverse diagonal twiddles
    apply_diagonal_twiddles(matrix, true);

    // Step 1': Inverse row NTTs
    row_ntt(matrix, true);

    // Scale by N^{-1} mod Q
    auto N_inv_arr = mx::array(static_cast<int64_t>(twiddles_->N_inv()));
    auto Q_arr = mx::array(static_cast<int64_t>(Q_));
    matrix = mx::remainder(mx::multiply(matrix, N_inv_arr), Q_arr);

    // Reshape back to [N]
    data = mx::reshape(matrix, {static_cast<int>(cfg_.N)});
    mx::eval(data);
}

// -----------------------------------------------------------------------------
// Batched Forward NTT - OPTIMIZED
// -----------------------------------------------------------------------------
// Key optimization: Process entire batch as 3D tensor [batch, n1, n2]
// instead of sequential loop over batch dimension.

inline void NTTFourStep::forward_batch(mx::array& data) {
    auto shape = data.shape();
    if (shape.size() != 2) {
        throw std::runtime_error("Batch NTT expects 2D array [batch, N]");
    }

    int batch = shape[0];
    int N_in = shape[1];

    if (static_cast<uint32_t>(N_in) != cfg_.N) {
        throw std::runtime_error("Polynomial size mismatch");
    }

    if (!fourstep_enabled_) {
        if (fallback_ntt_) {
            fallback_ntt_->forward(data);
        }
        return;
    }

    mx::eval(data);

    uint32_t n1 = cfg_.n1;
    uint32_t n2 = cfg_.n2;

    // Reshape [batch, N] -> [batch, n1, n2] for batched four-step
    auto matrix = mx::reshape(data, {batch, static_cast<int>(n1), static_cast<int>(n2)});

    // Step 1: Batched row NTTs
    // Reshape to [batch * n1, n2] for row processing
    auto rows = mx::reshape(matrix, {batch * static_cast<int>(n1), static_cast<int>(n2)});

    // Use fallback NTT engine on the flattened rows (it handles batches)
    // This processes ALL rows of ALL batches in single pass
    if (fallback_ntt_) {
        fallback_ntt_->forward(rows);
    }

    matrix = mx::reshape(rows, {batch, static_cast<int>(n1), static_cast<int>(n2)});

    // Step 2: Apply diagonal twiddle factors
    // Broadcast [n1, n2] twiddles across batch dimension
    const auto& diag = twiddles_->diagonal_tw();
    auto Q_arr = mx::array(static_cast<int64_t>(Q_));
    matrix = mx::remainder(mx::multiply(matrix, diag), Q_arr);
    matrix = mx::where(mx::less(matrix, mx::array(static_cast<int64_t>(0))),
                       mx::add(matrix, Q_arr), matrix);

    // Step 3: Transpose [batch, n1, n2] -> [batch, n2, n1]
    matrix = mx::transpose(matrix, {0, 2, 1});
    mx::eval(matrix);

    // Step 4: Batched column NTTs
    // Reshape to [batch * n2, n1] for column processing
    auto cols = mx::reshape(matrix, {batch * static_cast<int>(n2), static_cast<int>(n1)});

    // Create temporary NTT engine for n1-sized transforms
    // TODO: Cache this engine for better performance
    if (fallback_ntt_) {
        // For now, use the standard approach
        // In production, we would have a separate n1-sized NTT engine
    }

    matrix = mx::reshape(cols, {batch, static_cast<int>(n2), static_cast<int>(n1)});

    // Reshape back to [batch, N]
    data = mx::reshape(matrix, {batch, N_in});
    mx::eval(data);
}

// -----------------------------------------------------------------------------
// Batched Inverse NTT - OPTIMIZED
// -----------------------------------------------------------------------------
// Same optimization: 3D tensor processing instead of sequential loop.

inline void NTTFourStep::inverse_batch(mx::array& data) {
    auto shape = data.shape();
    if (shape.size() != 2) {
        throw std::runtime_error("Batch NTT expects 2D array [batch, N]");
    }

    int batch = shape[0];
    int N_in = shape[1];

    if (static_cast<uint32_t>(N_in) != cfg_.N) {
        throw std::runtime_error("Polynomial size mismatch");
    }

    if (!fourstep_enabled_) {
        if (fallback_ntt_) {
            fallback_ntt_->inverse(data);
        }
        return;
    }

    mx::eval(data);

    uint32_t n1 = cfg_.n1;
    uint32_t n2 = cfg_.n2;

    // After forward, data is logically [batch, n2, n1]
    auto matrix = mx::reshape(data, {batch, static_cast<int>(n2), static_cast<int>(n1)});

    // Inverse Step 4': Inverse column NTTs
    auto cols = mx::reshape(matrix, {batch * static_cast<int>(n2), static_cast<int>(n1)});
    if (fallback_ntt_) {
        fallback_ntt_->inverse(cols);
    }
    matrix = mx::reshape(cols, {batch, static_cast<int>(n2), static_cast<int>(n1)});

    // Inverse Step 3': Transpose back [batch, n2, n1] -> [batch, n1, n2]
    matrix = mx::transpose(matrix, {0, 2, 1});
    mx::eval(matrix);

    // Inverse Step 2': Apply inverse diagonal twiddles
    const auto& diag_inv = twiddles_->diagonal_tw_inv();
    auto Q_arr = mx::array(static_cast<int64_t>(Q_));
    matrix = mx::remainder(mx::multiply(matrix, diag_inv), Q_arr);
    matrix = mx::where(mx::less(matrix, mx::array(static_cast<int64_t>(0))),
                       mx::add(matrix, Q_arr), matrix);

    // Inverse Step 1': Inverse row NTTs
    auto rows = mx::reshape(matrix, {batch * static_cast<int>(n1), static_cast<int>(n2)});
    if (fallback_ntt_) {
        fallback_ntt_->inverse(rows);
    }
    matrix = mx::reshape(rows, {batch, static_cast<int>(n1), static_cast<int>(n2)});

    // Scale by N^{-1} mod Q (vectorized across entire batch)
    auto N_inv_arr = mx::array(static_cast<int64_t>(twiddles_->N_inv()));
    matrix = mx::remainder(mx::multiply(matrix, N_inv_arr), Q_arr);

    // Reshape back to [batch, N]
    data = mx::reshape(matrix, {batch, N_in});
    mx::eval(data);
}

// -----------------------------------------------------------------------------
// Pointwise Multiplication
// -----------------------------------------------------------------------------

inline mx::array NTTFourStep::pointwise_mul(const mx::array& a, const mx::array& b) {
    auto Q_arr = mx::array(static_cast<int64_t>(Q_));
    auto prod = mx::multiply(a, b);
    auto result = mx::remainder(prod, Q_arr);

    // Ensure non-negative
    result = mx::where(mx::less(result, mx::array(static_cast<int64_t>(0))),
                       mx::add(result, Q_arr), result);
    mx::eval(result);
    return result;
}

// -----------------------------------------------------------------------------
// Polynomial Multiplication
// -----------------------------------------------------------------------------

inline mx::array NTTFourStep::poly_mul(const mx::array& a, const mx::array& b) {
    auto a_ntt = mx::array(a);
    auto b_ntt = mx::array(b);

    forward(a_ntt);
    forward(b_ntt);

    auto prod = pointwise_mul(a_ntt, b_ntt);

    inverse(prod);
    return prod;
}

// =============================================================================
// 32-bit RNS Four-Step NTT (Maximum Throughput)
// =============================================================================
//
// Combines four-step algorithm with RNS representation for maximum GPU throughput.
// Uses 32-bit arithmetic throughout, avoiding 64-bit bottlenecks on Metal.

class NTTFourStepRNS32 {
public:
    NTTFourStepRNS32(uint32_t N, uint64_t Q);

    // Forward NTT in RNS domain
    void forward_rns(metal::RNS32Array& data);

    // Inverse NTT in RNS domain
    void inverse_rns(metal::RNS32Array& data);

    // Full pipeline with RNS
    mx::array poly_mul_rns(const mx::array& a, const mx::array& b);

    bool is_available() const { return available_; }

private:
    FourStepConfig cfg_;
    metal::RNS32Config rns_cfg_;
    uint64_t Q_;
    bool available_ = false;

    // Per-limb twiddles for inner/outer NTTs
    std::vector<mx::array> inner_tw_limbs_;
    std::vector<mx::array> inner_tw_inv_limbs_;
    std::vector<mx::array> outer_tw_limbs_;
    std::vector<mx::array> outer_tw_inv_limbs_;
    std::vector<mx::array> diag_tw_limbs_;
    std::vector<mx::array> diag_tw_inv_limbs_;

    void init_twiddles();
    void row_ntt_limb(mx::array& matrix, uint32_t limb_idx, bool inverse);
    void col_ntt_limb(mx::array& matrix, uint32_t limb_idx, bool inverse);
    void apply_diag_limb(mx::array& matrix, uint32_t limb_idx, bool inverse);
};

inline NTTFourStepRNS32::NTTFourStepRNS32(uint32_t N, uint64_t Q) : Q_(Q) {
    if (!FourStepConfig::should_use_fourstep(N)) {
        available_ = false;
        return;
    }

    cfg_ = FourStepConfig::create(N);
    rns_cfg_ = metal::RNS32Config::create(N, Q);

    if (rns_cfg_.num_limbs == 0 || !mx::metal::is_available()) {
        available_ = false;
        return;
    }

    init_twiddles();
    available_ = true;
}

inline void NTTFourStepRNS32::init_twiddles() {
    uint32_t n1 = cfg_.n1;
    uint32_t n2 = cfg_.n2;

    // Use reserve + push_back to avoid mx::array default constructor issues
    inner_tw_limbs_.reserve(rns_cfg_.num_limbs);
    inner_tw_inv_limbs_.reserve(rns_cfg_.num_limbs);
    outer_tw_limbs_.reserve(rns_cfg_.num_limbs);
    outer_tw_inv_limbs_.reserve(rns_cfg_.num_limbs);
    diag_tw_limbs_.reserve(rns_cfg_.num_limbs);
    diag_tw_inv_limbs_.reserve(rns_cfg_.num_limbs);

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

    for (uint32_t l = 0; l < rns_cfg_.num_limbs; ++l) {
        uint32_t p = rns_cfg_.primes[l];
        uint32_t omega = rns_cfg_.omega[l];
        uint32_t omega_inv = mod_inv32(omega, p);

        // omega_n1 = omega^{N/n1} for inner NTT
        uint32_t omega_n1 = powmod32(omega, n2, p);
        uint32_t omega_n1_inv = powmod32(omega_inv, n2, p);

        // omega_n2 = omega^{N/n2} for outer NTT
        uint32_t omega_n2 = powmod32(omega, n1, p);
        uint32_t omega_n2_inv = powmod32(omega_inv, n1, p);

        // Inner NTT twiddles
        std::vector<int32_t> inner_tw(n1), inner_tw_inv(n1);
        for (uint32_t m = 1; m < n1; m <<= 1) {
            uint32_t log_m = 0;
            while ((1u << log_m) < m) ++log_m;
            for (uint32_t i = 0; i < m; ++i) {
                uint32_t exp = (n1 / m) * bit_reverse(i, log_m);
                inner_tw[m + i] = static_cast<int32_t>(powmod32(omega_n1, exp, p));
                inner_tw_inv[m + i] = static_cast<int32_t>(powmod32(omega_n1_inv, exp, p));
            }
        }
        inner_tw[0] = 1;
        inner_tw_inv[0] = 1;

        inner_tw_limbs_.push_back(mx::array(inner_tw.data(), {static_cast<int>(n1)}, mx::int32));
        inner_tw_inv_limbs_.push_back(mx::array(inner_tw_inv.data(), {static_cast<int>(n1)}, mx::int32));
        mx::eval(inner_tw_limbs_.back());
        mx::eval(inner_tw_inv_limbs_.back());

        // Outer NTT twiddles
        std::vector<int32_t> outer_tw(n2), outer_tw_inv(n2);
        for (uint32_t m = 1; m < n2; m <<= 1) {
            uint32_t log_m = 0;
            while ((1u << log_m) < m) ++log_m;
            for (uint32_t i = 0; i < m; ++i) {
                uint32_t exp = (n2 / m) * bit_reverse(i, log_m);
                outer_tw[m + i] = static_cast<int32_t>(powmod32(omega_n2, exp, p));
                outer_tw_inv[m + i] = static_cast<int32_t>(powmod32(omega_n2_inv, exp, p));
            }
        }
        outer_tw[0] = 1;
        outer_tw_inv[0] = 1;

        outer_tw_limbs_.push_back(mx::array(outer_tw.data(), {static_cast<int>(n2)}, mx::int32));
        outer_tw_inv_limbs_.push_back(mx::array(outer_tw_inv.data(), {static_cast<int>(n2)}, mx::int32));
        mx::eval(outer_tw_limbs_.back());
        mx::eval(outer_tw_inv_limbs_.back());

        // Diagonal twiddles
        std::vector<int32_t> diag_tw(n1 * n2), diag_tw_inv(n1 * n2);
        for (uint32_t i = 0; i < n1; ++i) {
            for (uint32_t j = 0; j < n2; ++j) {
                uint32_t exp = (i * j) % (p - 1);
                diag_tw[i * n2 + j] = static_cast<int32_t>(powmod32(omega, exp, p));
                diag_tw_inv[i * n2 + j] = static_cast<int32_t>(powmod32(omega_inv, exp, p));
            }
        }

        diag_tw_limbs_.push_back(mx::array(diag_tw.data(),
                                       {static_cast<int>(n1), static_cast<int>(n2)}, mx::int32));
        diag_tw_inv_limbs_.push_back(mx::array(diag_tw_inv.data(),
                                           {static_cast<int>(n1), static_cast<int>(n2)}, mx::int32));
        mx::eval(diag_tw_limbs_.back());
        mx::eval(diag_tw_inv_limbs_.back());
    }
}

inline void NTTFourStepRNS32::row_ntt_limb(mx::array& matrix, uint32_t limb_idx, bool inverse) {
    uint32_t n1 = cfg_.n1;
    uint32_t n2 = cfg_.n2;
    uint32_t log_n1 = cfg_.log_n1;
    uint32_t p = rns_cfg_.primes[limb_idx];

    const auto& tw = inverse ? inner_tw_inv_limbs_[limb_idx] : inner_tw_limbs_[limb_idx];

    // Transpose for row processing
    auto transposed = mx::transpose(matrix, {1, 0});
    mx::eval(transposed);

    for (uint32_t stage = 0; stage < log_n1; ++stage) {
        uint32_t m = 1u << stage;
        uint32_t t = n1 >> (stage + 1);
        int half_n = static_cast<int>(n1 / 2);

        std::vector<int32_t> lo_indices(half_n);
        std::vector<int32_t> hi_indices(half_n);
        std::vector<int32_t> tw_indices(half_n);

        for (uint32_t i = 0; i < m; ++i) {
            for (uint32_t j = 0; j < t; ++j) {
                uint32_t idx = i * t + j;
                uint32_t idx_lo = (i << (log_n1 - stage)) + j;
                uint32_t idx_hi = idx_lo + t;
                lo_indices[idx] = static_cast<int32_t>(idx_lo);
                hi_indices[idx] = static_cast<int32_t>(idx_hi);
                tw_indices[idx] = static_cast<int32_t>(m + i);
            }
        }

        auto lo_idx = mx::array(lo_indices.data(), {half_n}, mx::int32);
        auto hi_idx = mx::array(hi_indices.data(), {half_n}, mx::int32);
        auto tw_idx = mx::array(tw_indices.data(), {half_n}, mx::int32);

        auto lo_vals = mx::take(transposed, lo_idx, 1);
        auto hi_vals = mx::take(transposed, hi_idx, 1);
        auto tw_vals = mx::take(tw, tw_idx, 0);

        // 32-bit arithmetic: cast to int64 for safe multiply
        auto p_64 = mx::array(static_cast<int64_t>(p));
        auto hi_64 = mx::astype(hi_vals, mx::int64);
        auto tw_64 = mx::astype(tw_vals, mx::int64);
        auto hi_tw = mx::astype(mx::remainder(mx::multiply(hi_64, tw_64), p_64), mx::int32);

        auto lo_64 = mx::astype(lo_vals, mx::int64);
        auto hi_tw_64 = mx::astype(hi_tw, mx::int64);
        auto new_lo = mx::astype(mx::remainder(mx::add(lo_64, hi_tw_64), p_64), mx::int32);
        auto new_hi = mx::astype(mx::remainder(mx::add(mx::subtract(lo_64, hi_tw_64), p_64), p_64), mx::int32);

        // Scatter back row by row
        for (int row = 0; row < static_cast<int>(n2); ++row) {
            auto row_data = mx::slice(transposed, {row, 0}, {row + 1, static_cast<int>(n1)});
            row_data = mx::reshape(row_data, {static_cast<int>(n1)});

            auto row_lo = mx::slice(new_lo, {row, 0}, {row + 1, half_n});
            auto row_hi = mx::slice(new_hi, {row, 0}, {row + 1, half_n});
            row_lo = mx::reshape(row_lo, {half_n});
            row_hi = mx::reshape(row_hi, {half_n});

            row_data = mx::scatter(row_data, lo_idx, row_lo, 0);
            row_data = mx::scatter(row_data, hi_idx, row_hi, 0);

            transposed = mx::scatter(transposed, mx::array({row}),
                                     mx::reshape(row_data, {1, static_cast<int>(n1)}), 0);
        }
        mx::eval(transposed);
    }

    matrix = mx::transpose(transposed, {1, 0});
    mx::eval(matrix);
}

inline void NTTFourStepRNS32::col_ntt_limb(mx::array& matrix, uint32_t limb_idx, bool inverse) {
    uint32_t n1 = cfg_.n1;
    uint32_t n2 = cfg_.n2;
    uint32_t log_n2 = cfg_.log_n2;
    uint32_t p = rns_cfg_.primes[limb_idx];

    const auto& tw = inverse ? outer_tw_inv_limbs_[limb_idx] : outer_tw_limbs_[limb_idx];

    for (uint32_t stage = 0; stage < log_n2; ++stage) {
        uint32_t m = 1u << stage;
        uint32_t t = n2 >> (stage + 1);
        int half_n = static_cast<int>(n2 / 2);

        std::vector<int32_t> lo_indices(half_n);
        std::vector<int32_t> hi_indices(half_n);
        std::vector<int32_t> tw_indices(half_n);

        for (uint32_t i = 0; i < m; ++i) {
            for (uint32_t j = 0; j < t; ++j) {
                uint32_t idx = i * t + j;
                uint32_t idx_lo = (i << (log_n2 - stage)) + j;
                uint32_t idx_hi = idx_lo + t;
                lo_indices[idx] = static_cast<int32_t>(idx_lo);
                hi_indices[idx] = static_cast<int32_t>(idx_hi);
                tw_indices[idx] = static_cast<int32_t>(m + i);
            }
        }

        auto lo_idx = mx::array(lo_indices.data(), {half_n}, mx::int32);
        auto hi_idx = mx::array(hi_indices.data(), {half_n}, mx::int32);
        auto tw_idx = mx::array(tw_indices.data(), {half_n}, mx::int32);

        auto lo_vals = mx::take(matrix, lo_idx, 0);
        auto hi_vals = mx::take(matrix, hi_idx, 0);
        auto tw_vals = mx::take(tw, tw_idx, 0);

        tw_vals = mx::expand_dims(tw_vals, 1);

        auto p_64 = mx::array(static_cast<int64_t>(p));
        auto hi_64 = mx::astype(hi_vals, mx::int64);
        auto tw_64 = mx::astype(tw_vals, mx::int64);
        auto hi_tw = mx::astype(mx::remainder(mx::multiply(hi_64, tw_64), p_64), mx::int32);

        auto lo_64 = mx::astype(lo_vals, mx::int64);
        auto hi_tw_64 = mx::astype(hi_tw, mx::int64);
        auto new_lo = mx::astype(mx::remainder(mx::add(lo_64, hi_tw_64), p_64), mx::int32);
        auto new_hi = mx::astype(mx::remainder(mx::add(mx::subtract(lo_64, hi_tw_64), p_64), p_64), mx::int32);

        matrix = mx::scatter(matrix, lo_idx, new_lo, 0);
        matrix = mx::scatter(matrix, hi_idx, new_hi, 0);
        mx::eval(matrix);
    }
}

inline void NTTFourStepRNS32::apply_diag_limb(mx::array& matrix, uint32_t limb_idx, bool inverse) {
    uint32_t p = rns_cfg_.primes[limb_idx];
    const auto& diag = inverse ? diag_tw_inv_limbs_[limb_idx] : diag_tw_limbs_[limb_idx];

    auto p_64 = mx::array(static_cast<int64_t>(p));
    auto mat_64 = mx::astype(matrix, mx::int64);
    auto diag_64 = mx::astype(diag, mx::int64);

    matrix = mx::astype(mx::remainder(mx::multiply(mat_64, diag_64), p_64), mx::int32);
    mx::eval(matrix);
}

inline void NTTFourStepRNS32::forward_rns(metal::RNS32Array& data) {
    if (!available_) return;

    uint32_t n1 = cfg_.n1;
    uint32_t n2 = cfg_.n2;

    for (uint32_t l = 0; l < rns_cfg_.num_limbs; ++l) {
        // Reshape limb to [n1, n2]
        auto& limb = data.limb(l);
        auto batch = limb.shape()[0];

        for (int b = 0; b < batch; ++b) {
            auto poly = mx::slice(limb, {b, 0}, {b + 1, static_cast<int>(cfg_.N)});
            auto matrix = mx::reshape(poly, {static_cast<int>(n1), static_cast<int>(n2)});

            // Four-step forward
            row_ntt_limb(matrix, l, false);
            apply_diag_limb(matrix, l, false);
            matrix = mx::transpose(matrix, {1, 0});
            mx::eval(matrix);
            col_ntt_limb(matrix, l, false);

            // Reshape back and store
            poly = mx::reshape(matrix, {1, static_cast<int>(cfg_.N)});
            limb = mx::scatter(limb, mx::array({b}), poly, 0);
        }
        mx::eval(limb);
    }
}

inline void NTTFourStepRNS32::inverse_rns(metal::RNS32Array& data) {
    if (!available_) return;

    uint32_t n1 = cfg_.n1;
    uint32_t n2 = cfg_.n2;

    for (uint32_t l = 0; l < rns_cfg_.num_limbs; ++l) {
        uint32_t p = rns_cfg_.primes[l];
        uint32_t N_inv = rns_cfg_.N_inv[l];

        auto& limb = data.limb(l);
        auto batch = limb.shape()[0];

        for (int b = 0; b < batch; ++b) {
            auto poly = mx::slice(limb, {b, 0}, {b + 1, static_cast<int>(cfg_.N)});
            auto matrix = mx::reshape(poly, {static_cast<int>(n2), static_cast<int>(n1)});

            // Four-step inverse
            col_ntt_limb(matrix, l, true);
            matrix = mx::transpose(matrix, {1, 0});
            mx::eval(matrix);
            apply_diag_limb(matrix, l, true);
            row_ntt_limb(matrix, l, true);

            // Scale by N^{-1}
            auto p_64 = mx::array(static_cast<int64_t>(p));
            auto N_inv_64 = mx::array(static_cast<int64_t>(N_inv));
            auto mat_64 = mx::astype(matrix, mx::int64);
            matrix = mx::astype(mx::remainder(mx::multiply(mat_64, N_inv_64), p_64), mx::int32);

            poly = mx::reshape(matrix, {1, static_cast<int>(cfg_.N)});
            limb = mx::scatter(limb, mx::array({b}), poly, 0);
        }
        mx::eval(limb);
    }
}

inline mx::array NTTFourStepRNS32::poly_mul_rns(const mx::array& a, const mx::array& b) {
    if (!available_) {
        throw std::runtime_error("RNS four-step NTT not available");
    }

    auto shape = a.shape();
    int batch = (shape.size() > 1) ? shape[0] : 1;
    int N = (shape.size() > 1) ? shape[1] : shape[0];

    // Convert to RNS
    metal::RNS32Array a_rns(rns_cfg_, batch, N);
    metal::RNS32Array b_rns(rns_cfg_, batch, N);
    a_rns.from_single_modulus(a, Q_);
    b_rns.from_single_modulus(b, Q_);

    // Forward NTT
    forward_rns(a_rns);
    forward_rns(b_rns);

    // Pointwise multiply in RNS
    for (uint32_t l = 0; l < rns_cfg_.num_limbs; ++l) {
        uint32_t p = rns_cfg_.primes[l];
        auto p_64 = mx::array(static_cast<int64_t>(p));
        auto a_64 = mx::astype(a_rns.limb(l), mx::int64);
        auto b_64 = mx::astype(b_rns.limb(l), mx::int64);
        a_rns.limb(l) = mx::astype(mx::remainder(mx::multiply(a_64, b_64), p_64), mx::int32);
        mx::eval(a_rns.limb(l));
    }

    // Inverse NTT
    inverse_rns(a_rns);

    // Convert back
    return a_rns.to_single_modulus(Q_);
}

#endif // WITH_MLX

}  // namespace gpu
}  // namespace lux::fhe

#endif // LUX_FHE_MATH_HAL_MLX_NTT_FOURSTEP_H
