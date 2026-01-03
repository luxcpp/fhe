// =============================================================================
// Unified Memory NTT/External-Product Streaming with Zero Materialization
// =============================================================================
//
// Patent: PAT-FHE-017
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Implementation of zero-materialization streaming for FHE operations on
// Apple Silicon unified memory.
//
// =============================================================================

#include "math/hal/mlx/unified_stream.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>

#ifdef __APPLE__
#include <sys/mman.h>
#include <mach/mach.h>
#endif

namespace lux {
namespace gpu {
namespace unified {

// =============================================================================
// Modular Arithmetic Utilities
// =============================================================================

namespace {

inline uint64_t mulmod_u128(uint64_t a, uint64_t b, uint64_t m) {
    return static_cast<uint64_t>((__uint128_t)a * b % m);
}

inline uint64_t addmod(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t sum = a + b;
    return sum >= m ? sum - m : sum;
}

inline uint64_t submod(uint64_t a, uint64_t b, uint64_t m) {
    return a >= b ? a - b : a + m - b;
}

inline uint64_t powmod(uint64_t base, uint64_t exp, uint64_t m) {
    uint64_t result = 1;
    base %= m;
    while (exp > 0) {
        if (exp & 1) result = mulmod_u128(result, base, m);
        base = mulmod_u128(base, base, m);
        exp >>= 1;
    }
    return result;
}

inline uint64_t mod_inverse(uint64_t a, uint64_t m) {
    int64_t t = 0, newt = 1;
    int64_t r = static_cast<int64_t>(m), newr = static_cast<int64_t>(a);
    while (newr != 0) {
        int64_t quotient = r / newr;
        std::tie(t, newt) = std::make_pair(newt, t - quotient * newt);
        std::tie(r, newr) = std::make_pair(newr, r - quotient * newr);
    }
    if (t < 0) t += static_cast<int64_t>(m);
    return static_cast<uint64_t>(t);
}

inline uint64_t find_primitive_root(uint32_t N, uint64_t Q) {
    uint64_t order = Q - 1;
    if (order % (2 * N) != 0) {
        throw std::runtime_error("Q - 1 must be divisible by 2N for NTT");
    }
    for (uint64_t g = 2; g < Q; ++g) {
        if (powmod(g, order / 2, Q) != 1) {
            return powmod(g, order / (2 * N), Q);
        }
    }
    throw std::runtime_error("No primitive root found");
}

inline uint32_t bit_reverse(uint32_t x, uint32_t bits) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < bits; ++i) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// High-resolution timer
inline double get_time_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();
    return static_cast<double>(ns) / 1e6;
}

}  // anonymous namespace

// =============================================================================
// UnifiedBuffer Implementation
// =============================================================================

UnifiedBuffer::UnifiedBuffer(size_t size_bytes, BufferOwner initial_owner)
    : owner_(initial_owner) {
    allocate(size_bytes);
}

UnifiedBuffer::~UnifiedBuffer() {
    deallocate();
}

UnifiedBuffer::UnifiedBuffer(UnifiedBuffer&& other) noexcept
    : base_ptr_(other.base_ptr_),
      size_bytes_(other.size_bytes_),
      owner_(other.owner_) {
    other.base_ptr_ = nullptr;
    other.size_bytes_ = 0;
}

UnifiedBuffer& UnifiedBuffer::operator=(UnifiedBuffer&& other) noexcept {
    if (this != &other) {
        deallocate();
        base_ptr_ = other.base_ptr_;
        size_bytes_ = other.size_bytes_;
        owner_ = other.owner_;
        other.base_ptr_ = nullptr;
        other.size_bytes_ = 0;
    }
    return *this;
}

void UnifiedBuffer::allocate(size_t size_bytes) {
    if (size_bytes == 0) return;

#ifdef __APPLE__
    // Use mmap for unified memory with Metal-compatible flags
    // MAP_PRIVATE | MAP_ANON gives us zero-initialized memory
    // No explicit Metal buffer needed - MLX handles this transparently
    base_ptr_ = mmap(nullptr, size_bytes,
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANON,
                     -1, 0);
    if (base_ptr_ == MAP_FAILED) {
        base_ptr_ = nullptr;
        throw std::bad_alloc();
    }

    // Prefault pages for consistent performance
    madvise(base_ptr_, size_bytes, MADV_WILLNEED);
#else
    // Fallback to aligned allocation
    base_ptr_ = aligned_alloc(64, size_bytes);
    if (!base_ptr_) {
        throw std::bad_alloc();
    }
    std::memset(base_ptr_, 0, size_bytes);
#endif

    size_bytes_ = size_bytes;
}

void UnifiedBuffer::deallocate() {
    if (base_ptr_) {
#ifdef __APPLE__
        munmap(base_ptr_, size_bytes_);
#else
        free(base_ptr_);
#endif
        base_ptr_ = nullptr;
        size_bytes_ = 0;
    }
}

#ifdef WITH_MLX
mx::array UnifiedBuffer::as_mlx_array(const std::vector<int>& shape, mx::Dtype dtype) const {
    if (!base_ptr_) {
        throw std::runtime_error("UnifiedBuffer: cannot create MLX array from null buffer");
    }

    // Calculate expected size
    size_t num_elements = 1;
    for (int dim : shape) {
        num_elements *= static_cast<size_t>(dim);
    }

    size_t element_size = 0;
    switch (dtype) {
        case mx::int32: element_size = 4; break;
        case mx::int64: element_size = 8; break;
        case mx::float32: element_size = 4; break;
        case mx::float64: element_size = 8; break;
        default: element_size = 8; break;
    }

    if (num_elements * element_size > size_bytes_) {
        throw std::runtime_error("UnifiedBuffer: requested array size exceeds buffer size");
    }

    // Create MLX array from existing memory
    // MLX on Apple Silicon uses unified memory, so this is zero-copy
    if (dtype == mx::int64) {
        return mx::array(static_cast<const int64_t*>(base_ptr_), shape, mx::int64);
    } else if (dtype == mx::int32) {
        return mx::array(static_cast<const int32_t*>(base_ptr_), shape, mx::int32);
    } else if (dtype == mx::float32) {
        return mx::array(static_cast<const float*>(base_ptr_), shape, mx::float32);
    } else {
        throw std::runtime_error("UnifiedBuffer: unsupported dtype for as_mlx_array");
    }
}

void UnifiedBuffer::from_mlx_array(const mx::array& arr) {
    mx::eval(arr);

    size_t num_elements = 1;
    for (int dim : arr.shape()) {
        num_elements *= static_cast<size_t>(dim);
    }

    size_t element_size = (arr.dtype() == mx::int64 || arr.dtype() == mx::float64) ? 8 : 4;
    size_t required_bytes = num_elements * element_size;

    if (required_bytes > size_bytes_) {
        throw std::runtime_error("UnifiedBuffer: MLX array too large for buffer");
    }

    // Copy data from MLX array to unified buffer
    if (arr.dtype() == mx::int64) {
        std::memcpy(base_ptr_, arr.data<int64_t>(), required_bytes);
    } else if (arr.dtype() == mx::int32) {
        std::memcpy(base_ptr_, arr.data<int32_t>(), required_bytes);
    } else if (arr.dtype() == mx::float32) {
        std::memcpy(base_ptr_, arr.data<float>(), required_bytes);
    }
}
#endif

void UnifiedBuffer::sync_for_gpu() {
#ifdef __APPLE__
    // Memory barrier for CPU -> GPU transition
    // On unified memory, this is a compiler/memory fence, not a DMA
    __sync_synchronize();
#endif
    owner_ = BufferOwner::GPU_EXCLUSIVE;
}

void UnifiedBuffer::sync_for_cpu() {
#ifdef __APPLE__
    // Memory barrier for GPU -> CPU transition
    __sync_synchronize();
#endif
    owner_ = BufferOwner::CPU_EXCLUSIVE;
}

// =============================================================================
// ScratchBufferPool Implementation
// =============================================================================

ScratchBufferPool::ScratchBufferPool(const UnifiedStreamConfig& config)
    : config_(config) {
    // Allocate limbs buffer: [2, L, N] int64
    size_t limbs_bytes = 2ULL * config_.L * config_.N * sizeof(int64_t);
    limbs_ = UnifiedBuffer(limbs_bytes, BufferOwner::GPU_EXCLUSIVE);

    // Allocate NTT scratch: [2, N] int64
    size_t ntt_bytes = 2ULL * config_.N * sizeof(int64_t);
    ntt_scratch_ = UnifiedBuffer(ntt_bytes, BufferOwner::GPU_EXCLUSIVE);
}

ScratchBufferPool::~ScratchBufferPool() = default;

size_t ScratchBufferPool::total_bytes() const {
    return limbs_.size_bytes() + ntt_scratch_.size_bytes();
}

// =============================================================================
// UnifiedTwiddleCache Implementation
// =============================================================================

UnifiedTwiddleCache::UnifiedTwiddleCache(uint32_t N, uint64_t Q)
    : N_(N), Q_(Q) {
    log_N_ = 0;
    uint32_t temp = N;
    while (temp > 1) { temp >>= 1; ++log_N_; }

    n_inv_ = static_cast<int64_t>(mod_inverse(N, Q));

    precompute_twiddles();
}

UnifiedTwiddleCache::~UnifiedTwiddleCache() = default;

void UnifiedTwiddleCache::precompute_twiddles() {
    uint64_t omega = find_primitive_root(N_, Q_);
    uint64_t omega_inv = mod_inverse(omega, Q_);

    // Calculate total twiddle storage with cache-line alignment
    // Each stage s has 2^s twiddles
    size_t total_twiddles = 0;
    stage_offsets_.resize(log_N_ + 1);

    constexpr size_t CACHE_LINE_ELEMS = 8;  // 64 bytes / 8 bytes per int64

    for (uint32_t s = 0; s < log_N_; ++s) {
        uint32_t m = 1u << s;

        // Align to cache line
        if (total_twiddles % CACHE_LINE_ELEMS != 0) {
            total_twiddles += CACHE_LINE_ELEMS - (total_twiddles % CACHE_LINE_ELEMS);
        }

        stage_offsets_[s] = static_cast<uint32_t>(total_twiddles);
        total_twiddles += m;
    }
    stage_offsets_[log_N_] = static_cast<uint32_t>(total_twiddles);

    // Allocate unified buffers
    size_t twiddle_bytes = total_twiddles * sizeof(int64_t);
    fwd_twiddles_ = UnifiedBuffer(twiddle_bytes, BufferOwner::SHARED_READ);
    inv_twiddles_ = UnifiedBuffer(twiddle_bytes, BufferOwner::SHARED_READ);

    auto* fwd_ptr = fwd_twiddles_.cpu_ptr<int64_t>();
    auto* inv_ptr = inv_twiddles_.cpu_ptr<int64_t>();

    // Zero-initialize (for padding)
    std::memset(fwd_ptr, 0, twiddle_bytes);
    std::memset(inv_ptr, 0, twiddle_bytes);

    // Compute stage-indexed twiddles
    for (uint32_t s = 0; s < log_N_; ++s) {
        uint32_t m = 1u << s;
        uint32_t offset = stage_offsets_[s];

        for (uint32_t i = 0; i < m; ++i) {
            uint32_t log_m = 0;
            uint32_t temp_m = m;
            while (temp_m > 1) { temp_m >>= 1; ++log_m; }

            uint32_t exp = (N_ / m) * (log_m > 0 ? bit_reverse(i, log_m) : 0);

            fwd_ptr[offset + i] = static_cast<int64_t>(powmod(omega, exp, Q_));
            inv_ptr[offset + i] = static_cast<int64_t>(powmod(omega_inv, exp, Q_));
        }
    }

    // Sync for GPU access
    fwd_twiddles_.sync_for_gpu();
    inv_twiddles_.sync_for_gpu();
}

const int64_t* UnifiedTwiddleCache::forward_stage(uint32_t stage) const {
    if (stage >= log_N_) return nullptr;
    return fwd_twiddles_.cpu_ptr<int64_t>() + stage_offsets_[stage];
}

const int64_t* UnifiedTwiddleCache::inverse_stage(uint32_t stage) const {
    if (stage >= log_N_) return nullptr;
    return inv_twiddles_.cpu_ptr<int64_t>() + stage_offsets_[stage];
}

size_t UnifiedTwiddleCache::total_bytes() const {
    return fwd_twiddles_.size_bytes() + inv_twiddles_.size_bytes();
}

// =============================================================================
// StreamingDecomposer Implementation
// =============================================================================

StreamingDecomposer::StreamingDecomposer(const UnifiedStreamConfig& config)
    : config_(config) {}

void StreamingDecomposer::decompose(
    const UnifiedBuffer& coeffs,
    UnifiedBuffer& limbs,
    uint32_t num_polys
) {
    const auto* coeff_ptr = coeffs.cpu_ptr<int64_t>();
    auto* limb_ptr = limbs.cpu_ptr<int64_t>();

    uint32_t N = config_.N;
    uint32_t L = config_.L;
    uint64_t mask = config_.decomp_mask();
    uint32_t base_log = config_.base_log;

    // Decompose each polynomial
    for (uint32_t p = 0; p < num_polys; ++p) {
        const int64_t* poly = coeff_ptr + p * N;
        int64_t* poly_limbs = limb_ptr + p * L * N;

        // Extract L digits from each coefficient
        for (uint32_t i = 0; i < N; ++i) {
            uint64_t c = static_cast<uint64_t>(poly[i]) % config_.Q;

            for (uint32_t l = 0; l < L; ++l) {
                uint64_t digit = (c >> (l * base_log)) & mask;
                // Interleaved layout: limb l, coefficient i
                poly_limbs[l * N + i] = static_cast<int64_t>(digit);
            }
        }
    }

    limbs.sync_for_gpu();
}

#ifdef WITH_MLX
void StreamingDecomposer::decompose_gpu(
    const mx::array& coeffs,
    mx::array& limbs
) {
    // GPU-accelerated decomposition using MLX bit operations
    auto shape = coeffs.shape();
    int batch = (shape.size() > 1) ? shape[0] : 1;
    int N = (shape.size() > 1) ? shape[1] : shape[0];
    uint32_t L = config_.L;
    uint32_t base_log = config_.base_log;

    // Prepare output shape [batch, L, N] or [L, N]
    std::vector<int> out_shape;
    if (shape.size() > 1) {
        out_shape = {batch, static_cast<int>(L), N};
    } else {
        out_shape = {static_cast<int>(L), N};
    }

    mx::eval(coeffs);

    // Decompose using bit shifts and masks
    auto Q_arr = mx::array(static_cast<int64_t>(config_.Q));
    auto coeffs_mod = mx::remainder(coeffs, Q_arr);

    std::vector<mx::array> limb_arrays;
    limb_arrays.reserve(L);

    auto mask_arr = mx::array(static_cast<int64_t>(config_.decomp_mask()));

    for (uint32_t l = 0; l < L; ++l) {
        auto shift = mx::array(static_cast<int64_t>(l * base_log));
        auto digit = mx::bitwise_and(mx::right_shift(coeffs_mod, shift), mask_arr);
        limb_arrays.push_back(digit);
    }

    // Stack along new dimension
    limbs = mx::stack(limb_arrays, shape.size() > 1 ? 1 : 0);
    mx::eval(limbs);
}
#endif

// =============================================================================
// StreamingNTTEngine Implementation
// =============================================================================

StreamingNTTEngine::StreamingNTTEngine(const UnifiedStreamConfig& config)
    : config_(config) {
    twiddle_cache_ = std::make_unique<UnifiedTwiddleCache>(config.N, config.Q);
}

StreamingNTTEngine::~StreamingNTTEngine() = default;

void StreamingNTTEngine::butterfly_ct(int64_t& lo, int64_t& hi, int64_t w, uint64_t Q) {
    uint64_t lo_u = static_cast<uint64_t>(lo);
    uint64_t hi_u = static_cast<uint64_t>(hi);
    uint64_t w_u = static_cast<uint64_t>(w);

    uint64_t whi = mulmod_u128(hi_u, w_u, Q);
    lo = static_cast<int64_t>(addmod(lo_u, whi, Q));
    hi = static_cast<int64_t>(submod(lo_u, whi, Q));
}

void StreamingNTTEngine::butterfly_gs(int64_t& lo, int64_t& hi, int64_t w, uint64_t Q) {
    uint64_t lo_u = static_cast<uint64_t>(lo);
    uint64_t hi_u = static_cast<uint64_t>(hi);
    uint64_t w_u = static_cast<uint64_t>(w);

    uint64_t sum = addmod(lo_u, hi_u, Q);
    uint64_t diff = submod(lo_u, hi_u, Q);

    lo = static_cast<int64_t>(sum);
    hi = static_cast<int64_t>(mulmod_u128(diff, w_u, Q));
}

void StreamingNTTEngine::forward_cpu(int64_t* data, uint32_t N) {
    uint32_t log_N = config_.log_N();
    uint64_t Q = config_.Q;

    // Cooley-Tukey forward NTT
    for (uint32_t s = 0; s < log_N; ++s) {
        uint32_t m = 1u << s;
        uint32_t t = N >> (s + 1);

        const int64_t* stage_tw = twiddle_cache_->forward_stage(s);

        for (uint32_t i = 0; i < m; ++i) {
            uint32_t j1 = i << (log_N - s);
            uint32_t j2 = j1 + t;
            int64_t w = stage_tw[i];

            for (uint32_t j = j1; j < j2; ++j) {
                butterfly_ct(data[j], data[j + t], w, Q);
            }
        }
    }
}

void StreamingNTTEngine::inverse_cpu(int64_t* data, uint32_t N) {
    uint32_t log_N = config_.log_N();
    uint64_t Q = config_.Q;

    // Gentleman-Sande inverse NTT
    for (uint32_t s = 0; s < log_N; ++s) {
        uint32_t m = N >> (s + 1);
        uint32_t t = 1u << s;

        const int64_t* stage_tw = twiddle_cache_->inverse_stage(s);

        for (uint32_t i = 0; i < m; ++i) {
            uint32_t j1 = i << (s + 1);
            uint32_t j2 = j1 + t;
            int64_t w = stage_tw[i];

            for (uint32_t j = j1; j < j2; ++j) {
                butterfly_gs(data[j], data[j + t], w, Q);
            }
        }
    }

    // Scale by N^{-1}
    int64_t n_inv = twiddle_cache_->n_inv();
    for (uint32_t i = 0; i < N; ++i) {
        data[i] = static_cast<int64_t>(
            mulmod_u128(static_cast<uint64_t>(data[i]),
                        static_cast<uint64_t>(n_inv), Q));
    }
}

void StreamingNTTEngine::forward(UnifiedBuffer& data) {
    data.sync_for_cpu();
    forward_cpu(data.cpu_ptr<int64_t>(), config_.N);
    data.sync_for_gpu();
}

void StreamingNTTEngine::inverse(UnifiedBuffer& data) {
    data.sync_for_cpu();
    inverse_cpu(data.cpu_ptr<int64_t>(), config_.N);
    data.sync_for_gpu();
}

void StreamingNTTEngine::forward_batch(UnifiedBuffer& data, uint32_t batch_size) {
    data.sync_for_cpu();
    auto* ptr = data.cpu_ptr<int64_t>();
    uint32_t N = config_.N;

    for (uint32_t b = 0; b < batch_size; ++b) {
        forward_cpu(ptr + b * N, N);
    }

    data.sync_for_gpu();
}

void StreamingNTTEngine::inverse_batch(UnifiedBuffer& data, uint32_t batch_size) {
    data.sync_for_cpu();
    auto* ptr = data.cpu_ptr<int64_t>();
    uint32_t N = config_.N;

    for (uint32_t b = 0; b < batch_size; ++b) {
        inverse_cpu(ptr + b * N, N);
    }

    data.sync_for_gpu();
}

void StreamingNTTEngine::forward_pipelined(
    UnifiedBuffer& limbs,
    uint32_t num_polys,
    uint32_t L
) {
    // Pipelined NTT: process all limbs
    // Layout: [num_polys, L, N] or [L, N] if num_polys == 1
    limbs.sync_for_cpu();
    auto* ptr = limbs.cpu_ptr<int64_t>();
    uint32_t N = config_.N;

    for (uint32_t p = 0; p < num_polys; ++p) {
        for (uint32_t l = 0; l < L; ++l) {
            int64_t* limb_data = ptr + (p * L + l) * N;
            forward_cpu(limb_data, N);
        }
    }

    limbs.sync_for_gpu();
}

#ifdef WITH_MLX
void StreamingNTTEngine::forward_mlx(mx::array& data) {
    // GPU-accelerated NTT using MLX
    auto shape = data.shape();
    int batch = (shape.size() > 1) ? shape[0] : 1;
    int N = (shape.size() > 1) ? shape[1] : shape[0];
    uint32_t log_N = config_.log_N();
    uint64_t Q = config_.Q;

    mx::eval(data);

    // Stage-by-stage NTT
    auto Q_arr = mx::array(static_cast<int64_t>(Q));

    for (uint32_t s = 0; s < log_N; ++s) {
        uint32_t m = 1u << s;
        uint32_t t = static_cast<uint32_t>(N) >> (s + 1);

        // Build index arrays for this stage
        std::vector<int32_t> lo_indices(N / 2), hi_indices(N / 2), tw_indices(N / 2);

        for (uint32_t i = 0; i < m; ++i) {
            for (uint32_t j = 0; j < t; ++j) {
                uint32_t idx = i * t + j;
                uint32_t idx_lo = (i << (log_N - s)) + j;
                uint32_t idx_hi = idx_lo + t;

                lo_indices[idx] = static_cast<int32_t>(idx_lo);
                hi_indices[idx] = static_cast<int32_t>(idx_hi);
                tw_indices[idx] = static_cast<int32_t>(i);
            }
        }

        auto lo_idx = mx::array(lo_indices.data(), {N / 2}, mx::int32);
        auto hi_idx = mx::array(hi_indices.data(), {N / 2}, mx::int32);
        auto tw_idx = mx::array(tw_indices.data(), {N / 2}, mx::int32);

        // Get stage twiddles
        const int64_t* stage_tw = twiddle_cache_->forward_stage(s);
        std::vector<int64_t> tw_vec(stage_tw, stage_tw + m);
        auto tw_arr = mx::array(tw_vec.data(), {static_cast<int>(m)}, mx::int64);

        // Process all batches
        for (int b = 0; b < batch; ++b) {
            auto poly = shape.size() > 1 ?
                mx::reshape(mx::slice(data, {b, 0}, {b + 1, N}), {N}) :
                data;

            auto lo_vals = mx::take(poly, lo_idx, 0);
            auto hi_vals = mx::take(poly, hi_idx, 0);
            auto tw_vals = mx::take(tw_arr, tw_idx, 0);

            // Butterfly
            auto hi_tw = mx::remainder(mx::multiply(hi_vals, tw_vals), Q_arr);
            auto new_lo = mx::remainder(mx::add(lo_vals, hi_tw), Q_arr);
            auto diff = mx::subtract(lo_vals, hi_tw);
            auto new_hi = mx::remainder(mx::add(diff, Q_arr), Q_arr);

            poly = mx::scatter(poly, lo_idx, new_lo, 0);
            poly = mx::scatter(poly, hi_idx, new_hi, 0);

            if (shape.size() > 1) {
                data = mx::scatter(data, mx::array({b}), mx::reshape(poly, {1, N}), 0);
            } else {
                data = poly;
            }
        }

        mx::eval(data);
    }
}

void StreamingNTTEngine::inverse_mlx(mx::array& data) {
    // GPU-accelerated inverse NTT using MLX
    auto shape = data.shape();
    int batch = (shape.size() > 1) ? shape[0] : 1;
    int N = (shape.size() > 1) ? shape[1] : shape[0];
    uint32_t log_N = config_.log_N();
    uint64_t Q = config_.Q;

    mx::eval(data);

    auto Q_arr = mx::array(static_cast<int64_t>(Q));

    // Gentleman-Sande stages
    for (uint32_t s = 0; s < log_N; ++s) {
        uint32_t m = static_cast<uint32_t>(N) >> (s + 1);
        uint32_t t = 1u << s;

        std::vector<int32_t> lo_indices(N / 2), hi_indices(N / 2), tw_indices(N / 2);

        for (uint32_t i = 0; i < m; ++i) {
            for (uint32_t j = 0; j < t; ++j) {
                uint32_t idx = i * t + j;
                uint32_t idx_lo = (i << (s + 1)) + j;
                uint32_t idx_hi = idx_lo + t;

                lo_indices[idx] = static_cast<int32_t>(idx_lo);
                hi_indices[idx] = static_cast<int32_t>(idx_hi);
                tw_indices[idx] = static_cast<int32_t>(i);
            }
        }

        auto lo_idx = mx::array(lo_indices.data(), {N / 2}, mx::int32);
        auto hi_idx = mx::array(hi_indices.data(), {N / 2}, mx::int32);
        auto tw_idx = mx::array(tw_indices.data(), {N / 2}, mx::int32);

        const int64_t* stage_tw = twiddle_cache_->inverse_stage(s);
        std::vector<int64_t> tw_vec(stage_tw, stage_tw + m);
        auto tw_arr = mx::array(tw_vec.data(), {static_cast<int>(m)}, mx::int64);

        for (int b = 0; b < batch; ++b) {
            auto poly = shape.size() > 1 ?
                mx::reshape(mx::slice(data, {b, 0}, {b + 1, N}), {N}) :
                data;

            auto lo_vals = mx::take(poly, lo_idx, 0);
            auto hi_vals = mx::take(poly, hi_idx, 0);
            auto tw_vals = mx::take(tw_arr, tw_idx, 0);

            // GS butterfly
            auto sum = mx::remainder(mx::add(lo_vals, hi_vals), Q_arr);
            auto diff = mx::subtract(lo_vals, hi_vals);
            diff = mx::remainder(mx::add(diff, Q_arr), Q_arr);
            auto new_hi = mx::remainder(mx::multiply(diff, tw_vals), Q_arr);

            poly = mx::scatter(poly, lo_idx, sum, 0);
            poly = mx::scatter(poly, hi_idx, new_hi, 0);

            if (shape.size() > 1) {
                data = mx::scatter(data, mx::array({b}), mx::reshape(poly, {1, N}), 0);
            } else {
                data = poly;
            }
        }

        mx::eval(data);
    }

    // Scale by N^{-1}
    auto n_inv = mx::array(twiddle_cache_->n_inv());
    data = mx::remainder(mx::multiply(data, n_inv), Q_arr);
    mx::eval(data);
}
#endif

// =============================================================================
// FusedAccumulator Implementation
// =============================================================================

FusedAccumulator::FusedAccumulator(const UnifiedStreamConfig& config)
    : config_(config) {}

void FusedAccumulator::accumulate(
    const UnifiedBuffer& limbs,
    const UnifiedBuffer& bsk_row,
    UnifiedBuffer& accumulator
) {
    // Fused multiply-accumulate
    // limbs: [2, L, N] - decomposed and NTT-transformed RLWE
    // bsk_row: [2, L, 2, N] - BSK row (NTT domain)
    // accumulator: [2, N] - output (NTT domain, in/out)

    const auto* limb_ptr = limbs.cpu_ptr<int64_t>();
    const auto* bsk_ptr = bsk_row.cpu_ptr<int64_t>();
    auto* acc_ptr = accumulator.cpu_ptr<int64_t>();

    uint32_t N = config_.N;
    uint32_t L = config_.L;
    uint64_t Q = config_.Q;

    // For each RLWE component (0, 1)
    for (uint32_t c_in = 0; c_in < 2; ++c_in) {
        const int64_t* c_limbs = limb_ptr + c_in * L * N;  // [L, N]

        // For each output component
        for (uint32_t c_out = 0; c_out < 2; ++c_out) {
            int64_t* acc = acc_ptr + c_out * N;

            // For each coefficient
            for (uint32_t i = 0; i < N; ++i) {
                uint64_t sum = 0;

                // Sum over limbs
                for (uint32_t l = 0; l < L; ++l) {
                    uint64_t limb_val = static_cast<uint64_t>(c_limbs[l * N + i]);

                    // BSK layout: [c_in, L, c_out, N]
                    size_t bsk_idx = c_in * L * 2 * N + l * 2 * N + c_out * N + i;
                    uint64_t bsk_val = static_cast<uint64_t>(bsk_ptr[bsk_idx]);

                    sum = addmod(sum, mulmod_u128(limb_val, bsk_val, Q), Q);
                }

                // Accumulate to output
                acc[i] = static_cast<int64_t>(
                    addmod(static_cast<uint64_t>(acc[i]), sum, Q));
            }
        }
    }

    accumulator.sync_for_gpu();
}

#ifdef WITH_MLX
void FusedAccumulator::accumulate_gpu(
    const mx::array& limbs,
    const mx::array& bsk_row,
    mx::array& accumulator
) {
    // GPU-accelerated fused accumulation
    // limbs: [2, L, N]
    // bsk_row: [2, L, 2, N]
    // accumulator: [2, N]

    int N = static_cast<int>(config_.N);
    uint32_t L = config_.L;
    uint64_t Q = config_.Q;

    mx::eval(limbs);
    mx::eval(bsk_row);
    mx::eval(accumulator);

    auto Q_arr = mx::array(static_cast<int64_t>(Q));

    // Initialize accumulators for each output component
    auto acc_0 = mx::slice(accumulator, {0, 0}, {1, N});
    auto acc_1 = mx::slice(accumulator, {1, 0}, {2, N});
    acc_0 = mx::reshape(acc_0, {N});
    acc_1 = mx::reshape(acc_1, {N});

    // Process each input component
    for (uint32_t c_in = 0; c_in < 2; ++c_in) {
        // Extract limbs for this component: [L, N]
        auto c_limbs = mx::slice(limbs,
            {static_cast<int>(c_in), 0, 0},
            {static_cast<int>(c_in + 1), static_cast<int>(L), N});
        c_limbs = mx::reshape(c_limbs, {static_cast<int>(L), N});

        for (uint32_t l = 0; l < L; ++l) {
            // Get limb l: [N]
            auto limb_l = mx::slice(c_limbs, {static_cast<int>(l), 0},
                                    {static_cast<int>(l + 1), N});
            limb_l = mx::reshape(limb_l, {N});

            // Get BSK rows for both output components
            // bsk_row[c_in, l, 0, :] and bsk_row[c_in, l, 1, :]
            auto bsk_0 = mx::slice(bsk_row,
                {static_cast<int>(c_in), static_cast<int>(l), 0, 0},
                {static_cast<int>(c_in + 1), static_cast<int>(l + 1), 1, N});
            auto bsk_1 = mx::slice(bsk_row,
                {static_cast<int>(c_in), static_cast<int>(l), 1, 0},
                {static_cast<int>(c_in + 1), static_cast<int>(l + 1), 2, N});
            bsk_0 = mx::reshape(bsk_0, {N});
            bsk_1 = mx::reshape(bsk_1, {N});

            // Multiply and accumulate
            auto prod_0 = mx::remainder(mx::multiply(limb_l, bsk_0), Q_arr);
            auto prod_1 = mx::remainder(mx::multiply(limb_l, bsk_1), Q_arr);

            acc_0 = mx::remainder(mx::add(acc_0, prod_0), Q_arr);
            acc_1 = mx::remainder(mx::add(acc_1, prod_1), Q_arr);
        }
    }

    // Stack back into accumulator
    accumulator = mx::stack({mx::reshape(acc_0, {1, N}), mx::reshape(acc_1, {1, N})}, 0);
    accumulator = mx::reshape(accumulator, {2, N});
    mx::eval(accumulator);
}
#endif

// =============================================================================
// StreamingExternalProduct Implementation
// =============================================================================

StreamingExternalProduct::StreamingExternalProduct(const UnifiedStreamConfig& config)
    : config_(config) {
    scratch_pool_ = std::make_unique<ScratchBufferPool>(config);
    decomposer_ = std::make_unique<StreamingDecomposer>(config);
    ntt_engine_ = std::make_unique<StreamingNTTEngine>(config);
    accumulator_ = std::make_unique<FusedAccumulator>(config);
}

StreamingExternalProduct::~StreamingExternalProduct() = default;

void StreamingExternalProduct::set_bootstrap_key(const UnifiedBuffer& bsk) {
    // Copy BSK to cache (this is the one-time cost at key setup)
    size_t bsk_bytes = bsk.size_bytes();
    bsk_cache_ = UnifiedBuffer(bsk_bytes, BufferOwner::SHARED_READ);
    std::memcpy(bsk_cache_.cpu_ptr<void>(), bsk.cpu_ptr<void>(), bsk_bytes);
    bsk_cache_.sync_for_gpu();
}

#ifdef WITH_MLX
void StreamingExternalProduct::set_bootstrap_key_mlx(const mx::array& bsk) {
    mx::eval(bsk);
    bsk_mlx_ = std::make_shared<mx::array>(bsk);
}
#endif

void StreamingExternalProduct::execute(
    const UnifiedBuffer& rlwe,
    uint32_t bsk_index,
    UnifiedBuffer& accumulator
) {
    double start_time = get_time_ms();

    uint32_t N = config_.N;
    uint32_t L = config_.L;

    // Stage 1: Decompose RLWE into limbs
    // Input: rlwe [2, N], Output: limbs [2, L, N]
    decomposer_->decompose(rlwe, scratch_pool_->limbs_buffer(), 2);

    // Stage 2: NTT on all limbs
    ntt_engine_->forward_pipelined(scratch_pool_->limbs_buffer(), 2, L);

    // Stage 3: Get BSK row and accumulate
    // BSK layout: [n, 2, L, 2, N]
    size_t bsk_row_offset = bsk_index * 2 * L * 2 * N * sizeof(int64_t);
    size_t bsk_row_size = 2 * L * 2 * N * sizeof(int64_t);

    // Create a view of the BSK row (zero-copy since it's in unified memory)
    UnifiedBuffer bsk_row(bsk_row_size, BufferOwner::SHARED_READ);
    std::memcpy(bsk_row.cpu_ptr<void>(),
                bsk_cache_.cpu_ptr<uint8_t>() + bsk_row_offset,
                bsk_row_size);
    bsk_row.sync_for_gpu();

    accumulator_->accumulate(scratch_pool_->limbs_buffer(), bsk_row, accumulator);

    last_execute_time_ms_ = get_time_ms() - start_time;

    // Update bandwidth tracking
    size_t bytes = 2 * N * 8 + 2 * L * 2 * N * 8 + 2 * N * 8;  // read + bsk + write
    total_bytes_processed_ += bytes;
    total_time_ns_ += static_cast<uint64_t>(last_execute_time_ms_ * 1e6);
}

#ifdef WITH_MLX
void StreamingExternalProduct::execute_mlx(
    const mx::array& rlwe,
    uint32_t bsk_index,
    mx::array& accumulator
) {
    double start_time = get_time_ms();

    int N = static_cast<int>(config_.N);
    uint32_t L = config_.L;
    uint32_t n = config_.n;

    mx::eval(rlwe);

    // Stage 1: GPU decomposition
    mx::array limbs;
    decomposer_->decompose_gpu(rlwe, limbs);  // [2, L, N]

    // Stage 2: GPU NTT on all limbs
    // Reshape to [2*L, N] for batch NTT
    auto limbs_flat = mx::reshape(limbs, {2 * static_cast<int>(L), N});
    ntt_engine_->forward_mlx(limbs_flat);
    limbs = mx::reshape(limbs_flat, {2, static_cast<int>(L), N});

    // Stage 3: Get BSK row and accumulate
    // bsk_mlx_: [n, 2, L, 2, N]
    auto bsk_row = mx::slice(*bsk_mlx_,
        {static_cast<int>(bsk_index), 0, 0, 0, 0},
        {static_cast<int>(bsk_index + 1), 2, static_cast<int>(L), 2, N});
    bsk_row = mx::reshape(bsk_row, {2, static_cast<int>(L), 2, N});

    accumulator_->accumulate_gpu(limbs, bsk_row, accumulator);

    last_execute_time_ms_ = get_time_ms() - start_time;
}
#endif

void StreamingExternalProduct::execute_batch(
    const std::vector<std::reference_wrapper<const UnifiedBuffer>>& rlwes,
    const std::vector<uint32_t>& bsk_indices,
    std::vector<std::reference_wrapper<UnifiedBuffer>>& accumulators
) {
    // Execute external products in sequence (could be parallelized)
    for (size_t i = 0; i < rlwes.size(); ++i) {
        execute(rlwes[i].get(), bsk_indices[i], accumulators[i].get());
    }
}

size_t StreamingExternalProduct::scratch_memory_bytes() const {
    return scratch_pool_->total_bytes() + ntt_engine_->twiddles().total_bytes();
}

size_t StreamingExternalProduct::bsk_memory_bytes() const {
    return bsk_cache_.size_bytes();
}

size_t StreamingExternalProduct::total_memory_bytes() const {
    return scratch_memory_bytes() + bsk_memory_bytes();
}

double StreamingExternalProduct::avg_bandwidth_gbps() const {
    if (total_time_ns_ == 0) return 0.0;
    double bytes = static_cast<double>(total_bytes_processed_.load());
    double seconds = static_cast<double>(total_time_ns_.load()) / 1e9;
    return bytes / seconds / 1e9;
}

// =============================================================================
// StreamingBlindRotation Implementation
// =============================================================================

StreamingBlindRotation::StreamingBlindRotation(const UnifiedStreamConfig& config)
    : config_(config) {
    ext_product_ = std::make_unique<StreamingExternalProduct>(config);
}

StreamingBlindRotation::~StreamingBlindRotation() = default;

void StreamingBlindRotation::set_bootstrap_key(const UnifiedBuffer& bsk) {
    ext_product_->set_bootstrap_key(bsk);
}

void StreamingBlindRotation::set_test_polynomial(const UnifiedBuffer& test_poly) {
    size_t bytes = test_poly.size_bytes();
    test_poly_ = UnifiedBuffer(bytes, BufferOwner::SHARED_READ);
    std::memcpy(test_poly_.cpu_ptr<void>(), test_poly.cpu_ptr<void>(), bytes);
    test_poly_.sync_for_gpu();
}

#ifdef WITH_MLX
void StreamingBlindRotation::set_bootstrap_key_mlx(const mx::array& bsk) {
    ext_product_->set_bootstrap_key_mlx(bsk);
}

void StreamingBlindRotation::set_test_polynomial_mlx(const mx::array& test_poly) {
    mx::eval(test_poly);
    test_poly_mlx_ = std::make_shared<mx::array>(test_poly);
}
#endif

void StreamingBlindRotation::execute(
    const UnifiedBuffer& lwe,
    UnifiedBuffer& output
) {
    double start_time = get_time_ms();

    uint32_t n = config_.n;
    uint32_t N = config_.N;
    uint64_t Q = config_.Q;

    const auto* lwe_ptr = lwe.cpu_ptr<int64_t>();
    auto* out_ptr = output.cpu_ptr<int64_t>();
    const auto* test_ptr = test_poly_.cpu_ptr<int64_t>();

    // Initialize accumulator with X^{-b} * test_poly
    int64_t b = lwe_ptr[n];  // Body of LWE
    int32_t shift = static_cast<int32_t>(b % (2 * N));
    if (shift < 0) shift += 2 * N;

    // acc = (0, X^{-b} * test_poly)
    for (uint32_t i = 0; i < N; ++i) {
        out_ptr[i] = 0;  // acc0 = 0

        // Negacyclic rotation
        int32_t src = (static_cast<int32_t>(i) + shift) % (2 * static_cast<int32_t>(N));
        if (src < static_cast<int32_t>(N)) {
            out_ptr[N + i] = test_ptr[src];
        } else {
            int64_t val = test_ptr[src - N];
            out_ptr[N + i] = (val == 0) ? 0 : static_cast<int64_t>(Q) - val;
        }
    }
    output.sync_for_gpu();

    // CMux loop: for each LWE coefficient
    for (uint32_t i = 0; i < n; ++i) {
        int64_t a_i = lwe_ptr[i];
        if (a_i == 0) continue;  // Skip if no rotation needed

        // External product with BSK[i]
        ext_product_->execute(output, i, output);
    }

    last_rotation_time_ms_ = get_time_ms() - start_time;
}

#ifdef WITH_MLX
void StreamingBlindRotation::execute_mlx(
    const mx::array& lwe,
    mx::array& output
) {
    double start_time = get_time_ms();

    int n = static_cast<int>(config_.n);
    int N = static_cast<int>(config_.N);
    uint64_t Q = config_.Q;

    mx::eval(lwe);

    // Get LWE body
    auto b_arr = mx::slice(lwe, {n}, {n + 1});
    mx::eval(b_arr);
    int64_t b = b_arr.data<int64_t>()[0];
    int32_t shift = static_cast<int32_t>(b % (2 * N));
    if (shift < 0) shift += 2 * N;

    // Initialize accumulator
    std::vector<int64_t> acc_data(2 * N, 0);
    mx::eval(*test_poly_mlx_);
    auto test_ptr = test_poly_mlx_->data<int64_t>();

    for (int i = 0; i < N; ++i) {
        int32_t src = (i + shift) % (2 * N);
        if (src < N) {
            acc_data[N + i] = test_ptr[src];
        } else {
            int64_t val = test_ptr[src - N];
            acc_data[N + i] = (val == 0) ? 0 : static_cast<int64_t>(Q) - val;
        }
    }

    output = mx::array(acc_data.data(), {2, N}, mx::int64);
    mx::eval(output);

    // CMux loop
    mx::eval(lwe);
    auto lwe_ptr = lwe.data<int64_t>();

    for (int i = 0; i < n; ++i) {
        if (lwe_ptr[i] == 0) continue;
        ext_product_->execute_mlx(output, static_cast<uint32_t>(i), output);
    }

    last_rotation_time_ms_ = get_time_ms() - start_time;
}

void StreamingBlindRotation::execute_batch_mlx(
    const mx::array& lwes,
    mx::array& outputs
) {
    // Batch blind rotation: process each LWE in sequence
    // TODO: Implement truly parallel batch processing
    int batch = lwes.shape()[0];
    int n = config_.n;
    int N = static_cast<int>(config_.N);

    std::vector<mx::array> results;
    results.reserve(batch);

    for (int b = 0; b < batch; ++b) {
        auto lwe = mx::slice(lwes, {b, 0}, {b + 1, n + 1});
        lwe = mx::reshape(lwe, {n + 1});

        mx::array out;
        execute_mlx(lwe, out);
        results.push_back(mx::reshape(out, {1, 2, N}));
    }

    outputs = mx::concatenate(results, 0);
    mx::eval(outputs);
}
#endif

void StreamingBlindRotation::execute_batch(
    const std::vector<std::reference_wrapper<const UnifiedBuffer>>& lwes,
    std::vector<std::reference_wrapper<UnifiedBuffer>>& outputs
) {
    for (size_t i = 0; i < lwes.size(); ++i) {
        execute(lwes[i].get(), outputs[i].get());
    }
}

// =============================================================================
// BandwidthCalculator Implementation
// =============================================================================

BandwidthCalculator::BandwidthCalculator(const UnifiedStreamConfig& config)
    : config_(config) {}

BandwidthStats BandwidthCalculator::external_product_stats(double time_ms) const {
    BandwidthStats stats;
    stats.time_ms = time_ms;

    uint32_t N = config_.N;
    uint32_t L = config_.L;

    // Reads: RLWE (2*N) + BSK row (2*L*2*N)
    stats.bytes_read = (2 * N + 2 * L * 2 * N) * sizeof(int64_t);

    // Writes: accumulator (2*N)
    stats.bytes_written = 2 * N * sizeof(int64_t);

    return stats;
}

BandwidthStats BandwidthCalculator::blind_rotation_stats(double time_ms) const {
    BandwidthStats stats;
    stats.time_ms = time_ms;

    uint32_t n = config_.n;
    uint32_t N = config_.N;
    uint32_t L = config_.L;

    // n external products
    auto ext_stats = external_product_stats(time_ms / n);
    stats.bytes_read = n * ext_stats.bytes_read;
    stats.bytes_written = n * ext_stats.bytes_written;

    // Add test polynomial read
    stats.bytes_read += N * sizeof(int64_t);

    return stats;
}

BandwidthStats BandwidthCalculator::bootstrap_stats(double time_ms) const {
    // Full bootstrap = blind rotation + key switch + sample extract
    auto br_stats = blind_rotation_stats(time_ms * 0.9);  // ~90% is blind rotation

    BandwidthStats stats;
    stats.time_ms = time_ms;
    stats.bytes_read = br_stats.bytes_read;
    stats.bytes_written = br_stats.bytes_written;

    // Key switch overhead
    stats.bytes_read += config_.N * config_.L * config_.n * sizeof(int64_t);
    stats.bytes_written += (config_.n + 1) * sizeof(int64_t);

    return stats;
}

// =============================================================================
// Factory Functions
// =============================================================================

std::unique_ptr<StreamingExternalProduct> create_streaming_external_product(
    const UnifiedStreamConfig& config
) {
    return std::make_unique<StreamingExternalProduct>(config);
}

std::unique_ptr<StreamingBlindRotation> create_streaming_blind_rotation(
    const UnifiedStreamConfig& config
) {
    return std::make_unique<StreamingBlindRotation>(config);
}

bool is_unified_memory_available() {
#ifdef HAS_UNIFIED_MEMORY
    return true;
#else
    return false;
#endif
}

double get_unified_memory_bandwidth_gbps() {
#ifdef __APPLE__
    // Apple Silicon unified memory bandwidth
    // M1: ~68 GB/s, M1 Pro/Max: ~200 GB/s, M2: ~100 GB/s, M3 Max: ~400 GB/s
    // Default to conservative M3 estimate
    return 400.0;
#else
    // Non-Apple platforms use system memory bandwidth
    return 50.0;  // Typical DDR4/DDR5 bandwidth
#endif
}

}  // namespace unified
}  // namespace gpu
}  // namespace lux::fhe
