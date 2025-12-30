// =============================================================================
// Four-Step NTT for Apple Metal - Optimal Large Transform Implementation
// =============================================================================
//
// Patent-pending algorithm: Row-column decomposition with 2 global barriers
//
// Key innovation: Split N-point NTT into sqrt(N) x sqrt(N) matrix
//   Phase 1: Row-wise NTTs (parallel, no sync needed within rows)
//   Phase 2: Twiddle multiply + Transpose (fused, single barrier)
//   Phase 3: Column-wise NTTs (parallel)
//   Total: 2 threadgroup barriers vs log(N) for traditional radix-2
//
// Target sizes: N=1024, 2048, 4096, 8192
// Hardware: Apple Silicon M1/M2/M3/M4 with unified memory
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

#ifndef LBCRYPTO_MATH_HAL_MLX_NTT_FOUR_STEP_H
#define LBCRYPTO_MATH_HAL_MLX_NTT_FOUR_STEP_H

#include <cstdint>
#include <vector>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <cstring>

#ifdef __APPLE__
#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>
#endif

namespace lbcrypto {
namespace metal {

// =============================================================================
// Configuration
// =============================================================================

struct FourStepNTTConfig {
    uint32_t N;           // Total transform size (must be power of 2)
    uint32_t n1;          // Row dimension (inner NTT size)
    uint32_t n2;          // Column dimension (outer NTT size)
    uint32_t log_n1;      // log2(n1)
    uint32_t log_n2;      // log2(n2)
    uint64_t Q;           // Prime modulus
    uint64_t mu;          // Barrett constant: floor(2^64 / Q)

    // Optimal factorization for GPU cache hierarchy
    // n1 = threadgroup size (fits in shared memory)
    // n2 = number of threadgroups (parallelism)
    static FourStepNTTConfig create(uint32_t N, uint64_t Q) {
        FourStepNTTConfig cfg;
        cfg.N = N;
        cfg.Q = Q;

        // Compute Barrett constant
        cfg.mu = compute_barrett_mu(Q);

        // Validate N is power of 2
        uint32_t log_N = 0;
        while ((1u << log_N) < N) ++log_N;
        if ((1u << log_N) != N) {
            throw std::runtime_error("N must be power of 2");
        }

        // Optimal split: balance n1 (shared memory) and n2 (parallelism)
        // For Apple Silicon: n1 <= 64 fits nicely in L1, larger uses L2
        switch (N) {
            case 1024:
                cfg.n1 = 32; cfg.n2 = 32;
                cfg.log_n1 = 5; cfg.log_n2 = 5;
                break;
            case 2048:
                cfg.n1 = 32; cfg.n2 = 64;
                cfg.log_n1 = 5; cfg.log_n2 = 6;
                break;
            case 4096:
                cfg.n1 = 64; cfg.n2 = 64;
                cfg.log_n1 = 6; cfg.log_n2 = 6;
                break;
            case 8192:
                cfg.n1 = 64; cfg.n2 = 128;
                cfg.log_n1 = 6; cfg.log_n2 = 7;
                break;
            case 16384:
                cfg.n1 = 128; cfg.n2 = 128;
                cfg.log_n1 = 7; cfg.log_n2 = 7;
                break;
            default:
                // General case: split evenly
                cfg.log_n1 = log_N / 2;
                cfg.log_n2 = log_N - cfg.log_n1;
                cfg.n1 = 1u << cfg.log_n1;
                cfg.n2 = 1u << cfg.log_n2;
        }

        return cfg;
    }

    static uint64_t compute_barrett_mu(uint64_t Q) {
        // mu = floor(2^64 / Q)
        // For Q < 2^63, this is safe
        __uint128_t numerator = (__uint128_t)1 << 64;
        return static_cast<uint64_t>(numerator / Q);
    }

    // Minimum N for four-step to be beneficial
    static constexpr uint32_t MIN_FOURSTEP_N = 1024;

    bool is_fourstep_beneficial() const {
        return N >= MIN_FOURSTEP_N;
    }
};

// =============================================================================
// NTT Parameters Buffer (matches Metal struct)
// =============================================================================

struct alignas(16) NTTParamsMetal {
    uint64_t Q;           // Prime modulus
    uint64_t mu;          // Barrett constant
    uint64_t N_inv;       // N^{-1} mod Q
    uint64_t N_inv_precon;// Barrett precon for N_inv
    uint32_t N;           // Ring dimension
    uint32_t log_N;       // log2(N)
    uint32_t n1;          // Row dimension
    uint32_t n2;          // Column dimension
    uint32_t log_n1;      // log2(n1)
    uint32_t log_n2;      // log2(n2)
    uint32_t _pad[2];     // Alignment padding
};

// =============================================================================
// Modular Arithmetic Helpers (CPU-side)
// =============================================================================

namespace detail {

inline uint64_t mod_mul_u128(uint64_t a, uint64_t b, uint64_t Q) {
    __uint128_t prod = (__uint128_t)a * b;
    return static_cast<uint64_t>(prod % Q);
}

inline uint64_t powmod(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) {
            result = mod_mul_u128(result, base, mod);
        }
        base = mod_mul_u128(base, base, mod);
        exp >>= 1;
    }
    return result;
}

inline uint64_t mod_inverse(uint64_t a, uint64_t m) {
    // Fermat's little theorem: a^{-1} = a^{m-2} mod m
    return powmod(a, m - 2, m);
}

inline uint32_t bit_reverse(uint32_t x, uint32_t log_n) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < log_n; ++i) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// Find primitive 2N-th root of unity
inline uint64_t find_primitive_root(uint32_t N, uint64_t Q) {
    uint64_t order = Q - 1;
    uint64_t target = 2 * static_cast<uint64_t>(N);

    if (order % target != 0) {
        throw std::runtime_error("Q-1 must be divisible by 2N");
    }

    // Find generator by trial
    for (uint64_t g = 2; g < Q; ++g) {
        // Quick check: g^{(Q-1)/2} should be Q-1 (-1 mod Q)
        if (powmod(g, order / 2, Q) != Q - 1) continue;

        // Compute 2N-th root
        uint64_t omega = powmod(g, order / target, Q);

        // Verify: omega^{2N} = 1 and omega^N != 1
        if (powmod(omega, target, Q) == 1 && powmod(omega, N, Q) != 1) {
            return omega;
        }
    }
    throw std::runtime_error("No primitive root found");
}

// Compute Barrett precomputation: floor(2^64 * x / Q)
inline uint64_t barrett_precon(uint64_t x, uint64_t Q) {
    __uint128_t numerator = (__uint128_t)x << 64;
    return static_cast<uint64_t>(numerator / Q);
}

} // namespace detail

// =============================================================================
// Twiddle Factor Tables
// =============================================================================

class FourStepTwiddles {
public:
    FourStepTwiddles(const FourStepNTTConfig& cfg)
        : cfg_(cfg), Q_(cfg.Q) {
        compute_twiddles();
    }

    // Accessors for GPU upload
    const std::vector<uint64_t>& row_twiddles() const { return row_tw_; }
    const std::vector<uint64_t>& row_twiddles_precon() const { return row_tw_precon_; }
    const std::vector<uint64_t>& row_inv_twiddles() const { return row_inv_tw_; }
    const std::vector<uint64_t>& row_inv_twiddles_precon() const { return row_inv_tw_precon_; }

    const std::vector<uint64_t>& col_twiddles() const { return col_tw_; }
    const std::vector<uint64_t>& col_twiddles_precon() const { return col_tw_precon_; }
    const std::vector<uint64_t>& col_inv_twiddles() const { return col_inv_tw_; }
    const std::vector<uint64_t>& col_inv_twiddles_precon() const { return col_inv_tw_precon_; }

    const std::vector<uint64_t>& diagonal_twiddles() const { return diag_tw_; }
    const std::vector<uint64_t>& diagonal_twiddles_precon() const { return diag_tw_precon_; }
    const std::vector<uint64_t>& diagonal_inv_twiddles() const { return diag_inv_tw_; }
    const std::vector<uint64_t>& diagonal_inv_twiddles_precon() const { return diag_inv_tw_precon_; }

    uint64_t N_inv() const { return N_inv_; }
    uint64_t N_inv_precon() const { return N_inv_precon_; }

private:
    FourStepNTTConfig cfg_;
    uint64_t Q_;
    uint64_t N_inv_;
    uint64_t N_inv_precon_;

    // Row NTT twiddles [n1] - for size-n1 NTTs within rows
    std::vector<uint64_t> row_tw_;
    std::vector<uint64_t> row_tw_precon_;
    std::vector<uint64_t> row_inv_tw_;
    std::vector<uint64_t> row_inv_tw_precon_;

    // Column NTT twiddles [n2] - for size-n2 NTTs within columns
    std::vector<uint64_t> col_tw_;
    std::vector<uint64_t> col_tw_precon_;
    std::vector<uint64_t> col_inv_tw_;
    std::vector<uint64_t> col_inv_tw_precon_;

    // Diagonal twiddles [n1 * n2] - omega^{i*j} for twist step
    std::vector<uint64_t> diag_tw_;
    std::vector<uint64_t> diag_tw_precon_;
    std::vector<uint64_t> diag_inv_tw_;
    std::vector<uint64_t> diag_inv_tw_precon_;

    void compute_twiddles() {
        uint32_t N = cfg_.N;
        uint32_t n1 = cfg_.n1;
        uint32_t n2 = cfg_.n2;

        // Find primitive 2N-th root of unity
        uint64_t omega = detail::find_primitive_root(N, Q_);
        uint64_t omega_inv = detail::mod_inverse(omega, Q_);

        // N^{-1} mod Q
        N_inv_ = detail::mod_inverse(N, Q_);
        N_inv_precon_ = detail::barrett_precon(N_inv_, Q_);

        // Row twiddles: omega_n1 = omega^{n2} (primitive 2n1-th root)
        uint64_t omega_n1 = detail::powmod(omega, n2, Q_);
        uint64_t omega_n1_inv = detail::powmod(omega_inv, n2, Q_);
        compute_ntt_twiddles(n1, omega_n1, omega_n1_inv, row_tw_, row_tw_precon_,
                             row_inv_tw_, row_inv_tw_precon_);

        // Column twiddles: omega_n2 = omega^{n1} (primitive 2n2-th root)
        uint64_t omega_n2 = detail::powmod(omega, n1, Q_);
        uint64_t omega_n2_inv = detail::powmod(omega_inv, n1, Q_);
        compute_ntt_twiddles(n2, omega_n2, omega_n2_inv, col_tw_, col_tw_precon_,
                             col_inv_tw_, col_inv_tw_precon_);

        // Diagonal twiddles: omega^{i*j} for 0 <= i < n1, 0 <= j < n2
        compute_diagonal_twiddles(n1, n2, omega, omega_inv);
    }

    void compute_ntt_twiddles(uint32_t n, uint64_t omega, uint64_t omega_inv,
                               std::vector<uint64_t>& tw, std::vector<uint64_t>& tw_precon,
                               std::vector<uint64_t>& inv_tw, std::vector<uint64_t>& inv_tw_precon) {
        tw.resize(n);
        tw_precon.resize(n);
        inv_tw.resize(n);
        inv_tw_precon.resize(n);

        uint32_t log_n = 0;
        while ((1u << log_n) < n) ++log_n;

        // OpenFHE-style bit-reversed twiddle storage
        // tw[m + i] = omega^{(n/m) * bitrev(i, log(m))} for stage m
        tw[0] = 1;
        tw_precon[0] = detail::barrett_precon(1, Q_);
        inv_tw[0] = 1;
        inv_tw_precon[0] = detail::barrett_precon(1, Q_);

        for (uint32_t m = 1; m < n; m <<= 1) {
            uint32_t log_m = 0;
            while ((1u << log_m) < m) ++log_m;

            for (uint32_t i = 0; i < m; ++i) {
                uint32_t exp = (n / m) * detail::bit_reverse(i, log_m);
                uint64_t w = detail::powmod(omega, exp, Q_);
                uint64_t w_inv = detail::powmod(omega_inv, exp, Q_);

                tw[m + i] = w;
                tw_precon[m + i] = detail::barrett_precon(w, Q_);
                inv_tw[m + i] = w_inv;
                inv_tw_precon[m + i] = detail::barrett_precon(w_inv, Q_);
            }
        }
    }

    void compute_diagonal_twiddles(uint32_t n1, uint32_t n2,
                                    uint64_t omega, uint64_t omega_inv) {
        size_t size = n1 * n2;
        diag_tw_.resize(size);
        diag_tw_precon_.resize(size);
        diag_inv_tw_.resize(size);
        diag_inv_tw_precon_.resize(size);

        // omega^{i*j} for twist step
        // Row i, column j: index = i * n2 + j
        for (uint32_t i = 0; i < n1; ++i) {
            for (uint32_t j = 0; j < n2; ++j) {
                uint64_t exp = static_cast<uint64_t>(i) * j;
                uint64_t w = detail::powmod(omega, exp, Q_);
                uint64_t w_inv = detail::powmod(omega_inv, exp, Q_);

                size_t idx = i * n2 + j;
                diag_tw_[idx] = w;
                diag_tw_precon_[idx] = detail::barrett_precon(w, Q_);
                diag_inv_tw_[idx] = w_inv;
                diag_inv_tw_precon_[idx] = detail::barrett_precon(w_inv, Q_);
            }
        }
    }
};

// =============================================================================
// FourStepNTT - Main Engine Class
// =============================================================================

#ifdef __APPLE__

class FourStepNTT {
public:
    FourStepNTT(uint32_t N, uint64_t Q);
    ~FourStepNTT();

    // Prevent copying
    FourStepNTT(const FourStepNTT&) = delete;
    FourStepNTT& operator=(const FourStepNTT&) = delete;

    // Forward NTT: coefficient domain -> evaluation domain
    // data: [batch, N] uint64_t buffer
    void forward(MTL::Buffer* data, uint32_t batch_size);

    // Inverse NTT: evaluation domain -> coefficient domain
    // data: [batch, N] uint64_t buffer
    void inverse(MTL::Buffer* data, uint32_t batch_size);

    // Fused forward: row NTT + twiddle + transpose + col NTT in minimal dispatches
    void forward_fused(MTL::Buffer* data, uint32_t batch_size);

    // Fused inverse: col NTT + transpose + twiddle + row NTT + scale
    void inverse_fused(MTL::Buffer* data, uint32_t batch_size);

    // Pointwise multiply mod Q
    void pointwise_mul(MTL::Buffer* out, MTL::Buffer* a, MTL::Buffer* b, uint32_t count);

    // Full polynomial multiplication: c = a * b mod (X^N + 1, Q)
    void poly_mul(MTL::Buffer* c, MTL::Buffer* a, MTL::Buffer* b, uint32_t batch_size);

    // Configuration access
    const FourStepNTTConfig& config() const { return cfg_; }
    bool is_available() const { return device_ != nullptr; }
    uint64_t modulus() const { return cfg_.Q; }

    // Memory management helpers
    MTL::Buffer* allocate_buffer(size_t size);
    void upload_data(MTL::Buffer* buffer, const uint64_t* data, size_t count);
    void download_data(uint64_t* data, MTL::Buffer* buffer, size_t count);

private:
    FourStepNTTConfig cfg_;
    std::unique_ptr<FourStepTwiddles> twiddles_;

    // Metal objects
    MTL::Device* device_ = nullptr;
    MTL::CommandQueue* queue_ = nullptr;
    MTL::Library* library_ = nullptr;

    // Compute pipelines
    MTL::ComputePipelineState* row_ntt_forward_ = nullptr;
    MTL::ComputePipelineState* row_ntt_inverse_ = nullptr;
    MTL::ComputePipelineState* col_ntt_forward_ = nullptr;
    MTL::ComputePipelineState* col_ntt_inverse_ = nullptr;
    MTL::ComputePipelineState* apply_diagonal_ = nullptr;
    MTL::ComputePipelineState* apply_diagonal_inv_ = nullptr;
    MTL::ComputePipelineState* transpose_kernel_ = nullptr;
    MTL::ComputePipelineState* scale_kernel_ = nullptr;
    MTL::ComputePipelineState* pointwise_mul_kernel_ = nullptr;

    // Fused kernels (2-barrier path)
    MTL::ComputePipelineState* fused_row_twiddle_transpose_ = nullptr;
    MTL::ComputePipelineState* fused_col_scale_ = nullptr;

    // GPU buffers for twiddles
    MTL::Buffer* row_tw_buf_ = nullptr;
    MTL::Buffer* row_tw_precon_buf_ = nullptr;
    MTL::Buffer* row_inv_tw_buf_ = nullptr;
    MTL::Buffer* row_inv_tw_precon_buf_ = nullptr;
    MTL::Buffer* col_tw_buf_ = nullptr;
    MTL::Buffer* col_tw_precon_buf_ = nullptr;
    MTL::Buffer* col_inv_tw_buf_ = nullptr;
    MTL::Buffer* col_inv_tw_precon_buf_ = nullptr;
    MTL::Buffer* diag_tw_buf_ = nullptr;
    MTL::Buffer* diag_tw_precon_buf_ = nullptr;
    MTL::Buffer* diag_inv_tw_buf_ = nullptr;
    MTL::Buffer* diag_inv_tw_precon_buf_ = nullptr;
    MTL::Buffer* params_buf_ = nullptr;

    void init_metal();
    void init_pipelines();
    void upload_twiddles();
    void cleanup();

    // Dispatch helpers
    void dispatch_row_ntt(MTL::CommandBuffer* cmd, MTL::Buffer* data,
                          uint32_t batch_size, bool inverse);
    void dispatch_col_ntt(MTL::CommandBuffer* cmd, MTL::Buffer* data,
                          uint32_t batch_size, bool inverse);
    void dispatch_diagonal(MTL::CommandBuffer* cmd, MTL::Buffer* data,
                           uint32_t batch_size, bool inverse);
    void dispatch_transpose(MTL::CommandBuffer* cmd, MTL::Buffer* data,
                            uint32_t batch_size);
    void dispatch_scale(MTL::CommandBuffer* cmd, MTL::Buffer* data,
                        uint32_t batch_size);
};

// =============================================================================
// Implementation
// =============================================================================

inline FourStepNTT::FourStepNTT(uint32_t N, uint64_t Q) {
    cfg_ = FourStepNTTConfig::create(N, Q);
    twiddles_ = std::make_unique<FourStepTwiddles>(cfg_);
    init_metal();
}

inline FourStepNTT::~FourStepNTT() {
    cleanup();
}

inline void FourStepNTT::init_metal() {
    // Get default Metal device
    device_ = MTL::CreateSystemDefaultDevice();
    if (!device_) {
        throw std::runtime_error("Metal device not available");
    }

    // Create command queue
    queue_ = device_->newCommandQueue();
    if (!queue_) {
        throw std::runtime_error("Failed to create command queue");
    }

    // Load shader library
    NS::Error* error = nullptr;
    NS::String* libPath = NS::String::string(
        "ntt_four_step.metallib", NS::UTF8StringEncoding);
    library_ = device_->newLibrary(libPath, &error);

    if (!library_) {
        // Try loading from source
        NS::String* srcPath = NS::String::string(
            "ntt_four_step.metal", NS::UTF8StringEncoding);
        MTL::CompileOptions* opts = MTL::CompileOptions::alloc()->init();
        library_ = device_->newLibrary(srcPath, opts, &error);
        opts->release();
    }

    if (!library_) {
        throw std::runtime_error("Failed to load Metal library");
    }

    init_pipelines();
    upload_twiddles();
}

inline void FourStepNTT::init_pipelines() {
    NS::Error* error = nullptr;

    auto create_pipeline = [&](const char* name) -> MTL::ComputePipelineState* {
        NS::String* fname = NS::String::string(name, NS::UTF8StringEncoding);
        MTL::Function* func = library_->newFunction(fname);
        if (!func) return nullptr;

        MTL::ComputePipelineState* pso = device_->newComputePipelineState(func, &error);
        func->release();
        return pso;
    };

    row_ntt_forward_ = create_pipeline("four_step_row_ntt_forward");
    row_ntt_inverse_ = create_pipeline("four_step_row_ntt_inverse");
    col_ntt_forward_ = create_pipeline("four_step_col_ntt_forward");
    col_ntt_inverse_ = create_pipeline("four_step_col_ntt_inverse");
    apply_diagonal_ = create_pipeline("four_step_apply_diagonal");
    apply_diagonal_inv_ = create_pipeline("four_step_apply_diagonal_inv");
    transpose_kernel_ = create_pipeline("four_step_transpose");
    scale_kernel_ = create_pipeline("four_step_scale");
    pointwise_mul_kernel_ = create_pipeline("four_step_pointwise_mul");
    fused_row_twiddle_transpose_ = create_pipeline("four_step_fused_row_twiddle_transpose");
    fused_col_scale_ = create_pipeline("four_step_fused_col_scale");
}

inline void FourStepNTT::upload_twiddles() {
    auto upload = [&](const std::vector<uint64_t>& data) -> MTL::Buffer* {
        size_t size = data.size() * sizeof(uint64_t);
        MTL::Buffer* buf = device_->newBuffer(size, MTL::ResourceStorageModeShared);
        std::memcpy(buf->contents(), data.data(), size);
        return buf;
    };

    row_tw_buf_ = upload(twiddles_->row_twiddles());
    row_tw_precon_buf_ = upload(twiddles_->row_twiddles_precon());
    row_inv_tw_buf_ = upload(twiddles_->row_inv_twiddles());
    row_inv_tw_precon_buf_ = upload(twiddles_->row_inv_twiddles_precon());
    col_tw_buf_ = upload(twiddles_->col_twiddles());
    col_tw_precon_buf_ = upload(twiddles_->col_twiddles_precon());
    col_inv_tw_buf_ = upload(twiddles_->col_inv_twiddles());
    col_inv_tw_precon_buf_ = upload(twiddles_->col_inv_twiddles_precon());
    diag_tw_buf_ = upload(twiddles_->diagonal_twiddles());
    diag_tw_precon_buf_ = upload(twiddles_->diagonal_twiddles_precon());
    diag_inv_tw_buf_ = upload(twiddles_->diagonal_inv_twiddles());
    diag_inv_tw_precon_buf_ = upload(twiddles_->diagonal_inv_twiddles_precon());

    // Parameters buffer
    NTTParamsMetal params;
    params.Q = cfg_.Q;
    params.mu = cfg_.mu;
    params.N_inv = twiddles_->N_inv();
    params.N_inv_precon = twiddles_->N_inv_precon();
    params.N = cfg_.N;
    params.log_N = cfg_.log_n1 + cfg_.log_n2;
    params.n1 = cfg_.n1;
    params.n2 = cfg_.n2;
    params.log_n1 = cfg_.log_n1;
    params.log_n2 = cfg_.log_n2;

    params_buf_ = device_->newBuffer(sizeof(params), MTL::ResourceStorageModeShared);
    std::memcpy(params_buf_->contents(), &params, sizeof(params));
}

inline void FourStepNTT::cleanup() {
    // Release pipelines
    if (row_ntt_forward_) row_ntt_forward_->release();
    if (row_ntt_inverse_) row_ntt_inverse_->release();
    if (col_ntt_forward_) col_ntt_forward_->release();
    if (col_ntt_inverse_) col_ntt_inverse_->release();
    if (apply_diagonal_) apply_diagonal_->release();
    if (apply_diagonal_inv_) apply_diagonal_inv_->release();
    if (transpose_kernel_) transpose_kernel_->release();
    if (scale_kernel_) scale_kernel_->release();
    if (pointwise_mul_kernel_) pointwise_mul_kernel_->release();
    if (fused_row_twiddle_transpose_) fused_row_twiddle_transpose_->release();
    if (fused_col_scale_) fused_col_scale_->release();

    // Release buffers
    if (row_tw_buf_) row_tw_buf_->release();
    if (row_tw_precon_buf_) row_tw_precon_buf_->release();
    if (row_inv_tw_buf_) row_inv_tw_buf_->release();
    if (row_inv_tw_precon_buf_) row_inv_tw_precon_buf_->release();
    if (col_tw_buf_) col_tw_buf_->release();
    if (col_tw_precon_buf_) col_tw_precon_buf_->release();
    if (col_inv_tw_buf_) col_inv_tw_buf_->release();
    if (col_inv_tw_precon_buf_) col_inv_tw_precon_buf_->release();
    if (diag_tw_buf_) diag_tw_buf_->release();
    if (diag_tw_precon_buf_) diag_tw_precon_buf_->release();
    if (diag_inv_tw_buf_) diag_inv_tw_buf_->release();
    if (diag_inv_tw_precon_buf_) diag_inv_tw_precon_buf_->release();
    if (params_buf_) params_buf_->release();

    // Release Metal objects
    if (library_) library_->release();
    if (queue_) queue_->release();
    if (device_) device_->release();
}

inline void FourStepNTT::forward(MTL::Buffer* data, uint32_t batch_size) {
    MTL::CommandBuffer* cmd = queue_->commandBuffer();

    // Phase 1: Row NTTs (n2 parallel NTTs of size n1)
    dispatch_row_ntt(cmd, data, batch_size, false);

    // Phase 2: Diagonal twiddle multiply
    dispatch_diagonal(cmd, data, batch_size, false);

    // Phase 3: Transpose (n1 x n2) -> (n2 x n1)
    dispatch_transpose(cmd, data, batch_size);

    // Phase 4: Column NTTs (n1 parallel NTTs of size n2)
    dispatch_col_ntt(cmd, data, batch_size, false);

    cmd->commit();
    cmd->waitUntilCompleted();
}

inline void FourStepNTT::inverse(MTL::Buffer* data, uint32_t batch_size) {
    MTL::CommandBuffer* cmd = queue_->commandBuffer();

    // Inverse steps in reverse order
    // Phase 1: Inverse column NTTs
    dispatch_col_ntt(cmd, data, batch_size, true);

    // Phase 2: Transpose (n2 x n1) -> (n1 x n2)
    dispatch_transpose(cmd, data, batch_size);

    // Phase 3: Inverse diagonal twiddle multiply
    dispatch_diagonal(cmd, data, batch_size, true);

    // Phase 4: Inverse row NTTs
    dispatch_row_ntt(cmd, data, batch_size, true);

    // Phase 5: Scale by N^{-1}
    dispatch_scale(cmd, data, batch_size);

    cmd->commit();
    cmd->waitUntilCompleted();
}

inline void FourStepNTT::forward_fused(MTL::Buffer* data, uint32_t batch_size) {
    // Optimal 2-barrier implementation:
    // Barrier 1: After row NTT + twiddle + transpose (fused)
    // Barrier 2: After column NTT

    MTL::CommandBuffer* cmd = queue_->commandBuffer();

    if (fused_row_twiddle_transpose_) {
        // Fused kernel: row_ntt -> twiddle -> transpose
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(fused_row_twiddle_transpose_);
        enc->setBuffer(data, 0, 0);
        enc->setBuffer(row_tw_buf_, 0, 1);
        enc->setBuffer(row_tw_precon_buf_, 0, 2);
        enc->setBuffer(diag_tw_buf_, 0, 3);
        enc->setBuffer(diag_tw_precon_buf_, 0, 4);
        enc->setBuffer(params_buf_, 0, 5);

        // Dispatch: one threadgroup per row, all rows in parallel
        MTL::Size threads(cfg_.n1, cfg_.n2, batch_size);
        MTL::Size threadgroup(cfg_.n1, 1, 1);
        enc->dispatchThreads(threads, threadgroup);
        enc->endEncoding();
    } else {
        // Fallback to separate dispatches
        dispatch_row_ntt(cmd, data, batch_size, false);
        dispatch_diagonal(cmd, data, batch_size, false);
        dispatch_transpose(cmd, data, batch_size);
    }

    // Column NTTs
    dispatch_col_ntt(cmd, data, batch_size, false);

    cmd->commit();
    cmd->waitUntilCompleted();
}

inline void FourStepNTT::inverse_fused(MTL::Buffer* data, uint32_t batch_size) {
    MTL::CommandBuffer* cmd = queue_->commandBuffer();

    // Inverse column NTTs
    dispatch_col_ntt(cmd, data, batch_size, true);

    if (fused_col_scale_) {
        // Fused: transpose -> twiddle_inv -> row_ntt_inv -> scale
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(fused_col_scale_);
        enc->setBuffer(data, 0, 0);
        enc->setBuffer(diag_inv_tw_buf_, 0, 1);
        enc->setBuffer(diag_inv_tw_precon_buf_, 0, 2);
        enc->setBuffer(row_inv_tw_buf_, 0, 3);
        enc->setBuffer(row_inv_tw_precon_buf_, 0, 4);
        enc->setBuffer(params_buf_, 0, 5);

        MTL::Size threads(cfg_.n1, cfg_.n2, batch_size);
        MTL::Size threadgroup(cfg_.n1, 1, 1);
        enc->dispatchThreads(threads, threadgroup);
        enc->endEncoding();
    } else {
        dispatch_transpose(cmd, data, batch_size);
        dispatch_diagonal(cmd, data, batch_size, true);
        dispatch_row_ntt(cmd, data, batch_size, true);
        dispatch_scale(cmd, data, batch_size);
    }

    cmd->commit();
    cmd->waitUntilCompleted();
}

inline void FourStepNTT::dispatch_row_ntt(MTL::CommandBuffer* cmd, MTL::Buffer* data,
                                           uint32_t batch_size, bool inverse) {
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
    MTL::ComputePipelineState* pso = inverse ? row_ntt_inverse_ : row_ntt_forward_;

    enc->setComputePipelineState(pso);
    enc->setBuffer(data, 0, 0);
    enc->setBuffer(inverse ? row_inv_tw_buf_ : row_tw_buf_, 0, 1);
    enc->setBuffer(inverse ? row_inv_tw_precon_buf_ : row_tw_precon_buf_, 0, 2);
    enc->setBuffer(params_buf_, 0, 3);

    // Each threadgroup handles one row (n1 elements)
    // n2 rows per polynomial, batch_size polynomials
    MTL::Size threads(cfg_.n1 / 2, cfg_.n2 * batch_size, 1);
    MTL::Size threadgroup(cfg_.n1 / 2, 1, 1);
    enc->dispatchThreads(threads, threadgroup);
    enc->endEncoding();
}

inline void FourStepNTT::dispatch_col_ntt(MTL::CommandBuffer* cmd, MTL::Buffer* data,
                                           uint32_t batch_size, bool inverse) {
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
    MTL::ComputePipelineState* pso = inverse ? col_ntt_inverse_ : col_ntt_forward_;

    enc->setComputePipelineState(pso);
    enc->setBuffer(data, 0, 0);
    enc->setBuffer(inverse ? col_inv_tw_buf_ : col_tw_buf_, 0, 1);
    enc->setBuffer(inverse ? col_inv_tw_precon_buf_ : col_tw_precon_buf_, 0, 2);
    enc->setBuffer(params_buf_, 0, 3);

    // Each threadgroup handles one column (n2 elements)
    MTL::Size threads(cfg_.n2 / 2, cfg_.n1 * batch_size, 1);
    MTL::Size threadgroup(cfg_.n2 / 2, 1, 1);
    enc->dispatchThreads(threads, threadgroup);
    enc->endEncoding();
}

inline void FourStepNTT::dispatch_diagonal(MTL::CommandBuffer* cmd, MTL::Buffer* data,
                                            uint32_t batch_size, bool inverse) {
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
    MTL::ComputePipelineState* pso = inverse ? apply_diagonal_inv_ : apply_diagonal_;

    enc->setComputePipelineState(pso);
    enc->setBuffer(data, 0, 0);
    enc->setBuffer(inverse ? diag_inv_tw_buf_ : diag_tw_buf_, 0, 1);
    enc->setBuffer(inverse ? diag_inv_tw_precon_buf_ : diag_tw_precon_buf_, 0, 2);
    enc->setBuffer(params_buf_, 0, 3);

    MTL::Size threads(cfg_.N, batch_size, 1);
    MTL::Size threadgroup(256, 1, 1);
    enc->dispatchThreads(threads, threadgroup);
    enc->endEncoding();
}

inline void FourStepNTT::dispatch_transpose(MTL::CommandBuffer* cmd, MTL::Buffer* data,
                                             uint32_t batch_size) {
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(transpose_kernel_);
    enc->setBuffer(data, 0, 0);
    enc->setBuffer(params_buf_, 0, 1);

    // Transpose n1 x n2 matrix
    MTL::Size threads(cfg_.n1, cfg_.n2, batch_size);
    MTL::Size threadgroup(16, 16, 1);
    enc->dispatchThreads(threads, threadgroup);
    enc->endEncoding();
}

inline void FourStepNTT::dispatch_scale(MTL::CommandBuffer* cmd, MTL::Buffer* data,
                                         uint32_t batch_size) {
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(scale_kernel_);
    enc->setBuffer(data, 0, 0);
    enc->setBuffer(params_buf_, 0, 1);

    MTL::Size threads(cfg_.N, batch_size, 1);
    MTL::Size threadgroup(256, 1, 1);
    enc->dispatchThreads(threads, threadgroup);
    enc->endEncoding();
}

inline void FourStepNTT::pointwise_mul(MTL::Buffer* out, MTL::Buffer* a,
                                        MTL::Buffer* b, uint32_t count) {
    MTL::CommandBuffer* cmd = queue_->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();

    enc->setComputePipelineState(pointwise_mul_kernel_);
    enc->setBuffer(out, 0, 0);
    enc->setBuffer(a, 0, 1);
    enc->setBuffer(b, 0, 2);
    enc->setBuffer(params_buf_, 0, 3);

    MTL::Size threads(count, 1, 1);
    MTL::Size threadgroup(256, 1, 1);
    enc->dispatchThreads(threads, threadgroup);
    enc->endEncoding();

    cmd->commit();
    cmd->waitUntilCompleted();
}

inline void FourStepNTT::poly_mul(MTL::Buffer* c, MTL::Buffer* a,
                                   MTL::Buffer* b, uint32_t batch_size) {
    // Forward NTT on both inputs
    forward(a, batch_size);
    forward(b, batch_size);

    // Pointwise multiply
    pointwise_mul(c, a, b, cfg_.N * batch_size);

    // Inverse NTT on result
    inverse(c, batch_size);
}

inline MTL::Buffer* FourStepNTT::allocate_buffer(size_t size) {
    return device_->newBuffer(size, MTL::ResourceStorageModeShared);
}

inline void FourStepNTT::upload_data(MTL::Buffer* buffer, const uint64_t* data, size_t count) {
    std::memcpy(buffer->contents(), data, count * sizeof(uint64_t));
}

inline void FourStepNTT::download_data(uint64_t* data, MTL::Buffer* buffer, size_t count) {
    std::memcpy(data, buffer->contents(), count * sizeof(uint64_t));
}

#endif // __APPLE__

} // namespace metal
} // namespace lbcrypto

#endif // LBCRYPTO_MATH_HAL_MLX_NTT_FOUR_STEP_H
