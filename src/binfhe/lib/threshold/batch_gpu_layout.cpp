// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// GPU-Optimized Memory Layout Implementation
//
// Key optimizations:
// 1. Cache-efficient tiled transpose (TILE_DIM x TILE_DIM blocks)
// 2. Vectorized inner product with amortized key loading
// 3. Memory alignment for coalesced GPU access
// 4. OpenMP parallelization for CPU fallback

#include "threshold/batch_gpu_layout.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace lbcrypto {
namespace threshold {

// ============================================================================
// LayoutParams Implementation
// ============================================================================

size_t LayoutParams::padded_batch() const {
    return RoundToWarp(batch_size);
}

size_t LayoutParams::padded_dim() const {
    return RoundToTile(dimension);
}

// ============================================================================
// BatchGPULayout Implementation
// ============================================================================

struct BatchGPULayout::Impl {
    LayoutParams params;
    NativeInteger modulus;

    // SoA storage: coeffs_a[coeff_index * batch_size + ct_index]
    std::vector<uint64_t> coeffs_a;

    // 'b' components: coeffs_b[ct_index]
    std::vector<uint64_t> coeffs_b;

    // Whether data is valid
    bool valid = false;

    Impl() = default;

    explicit Impl(const LayoutParams& p)
        : params(p), valid(false) {
        Allocate();
    }

    void Allocate() {
        size_t total_a = static_cast<size_t>(params.dimension) * params.batch_size;
        coeffs_a.resize(total_a, 0);
        coeffs_b.resize(params.batch_size, 0);
    }

    void Clear() {
        coeffs_a.clear();
        coeffs_b.clear();
        valid = false;
    }

    // SoA index calculation
    size_t SoAIndex(uint32_t ct_index, uint32_t coeff_index) const {
        // coeffs_a[coeff_index * batch_size + ct_index]
        return static_cast<size_t>(coeff_index) * params.batch_size + ct_index;
    }
};

BatchGPULayout::BatchGPULayout() : impl_(std::make_unique<Impl>()) {}

BatchGPULayout::BatchGPULayout(const LayoutParams& params)
    : impl_(std::make_unique<Impl>(params)) {}

BatchGPULayout::BatchGPULayout(const std::vector<LWECiphertext>& cts, MemoryLayout layout)
    : impl_(std::make_unique<Impl>()) {
    if (!cts.empty()) {
        impl_->params.batch_size = static_cast<uint32_t>(cts.size());
        impl_->params.dimension = cts[0]->GetLength();
        impl_->params.layout = layout;
        impl_->modulus = cts[0]->GetModulus();
        impl_->Allocate();
        ImportCiphertexts(cts);
    }
}

BatchGPULayout::~BatchGPULayout() = default;

BatchGPULayout::BatchGPULayout(BatchGPULayout&&) noexcept = default;
BatchGPULayout& BatchGPULayout::operator=(BatchGPULayout&&) noexcept = default;

// ============================================================================
// Data Import/Export
// ============================================================================

void BatchGPULayout::ImportCiphertexts(const std::vector<LWECiphertext>& cts) {
    if (cts.empty()) {
        impl_->Clear();
        return;
    }

    uint32_t batch_size = static_cast<uint32_t>(cts.size());
    uint32_t dimension = cts[0]->GetLength();

    // Resize if needed
    if (impl_->params.batch_size != batch_size || impl_->params.dimension != dimension) {
        impl_->params.batch_size = batch_size;
        impl_->params.dimension = dimension;
        impl_->Allocate();
    }

    impl_->modulus = cts[0]->GetModulus();

    // Convert from AoS (vector of ciphertexts) to SoA (coefficient slices)
    //
    // AoS: for each ct, iterate coefficients
    // SoA: for each coefficient index, iterate ciphertexts
    //
    // We do this in a cache-friendly manner using tiling.

    if (impl_->params.layout == MemoryLayout::SOA) {
        // Direct SoA import with tiling for cache efficiency
        //
        // Process in tiles to maximize cache reuse:
        // - Load TILE_DIM ciphertexts' coefficients into cache
        // - Write TILE_DIM coefficient slices

        const uint32_t tile = gpu::TILE_DIM;

        #pragma omp parallel for collapse(2) if(batch_size * dimension > 10000)
        for (uint32_t ct_tile = 0; ct_tile < batch_size; ct_tile += tile) {
            for (uint32_t coeff_tile = 0; coeff_tile < dimension; coeff_tile += tile) {
                // Process this tile
                uint32_t ct_end = std::min(ct_tile + tile, batch_size);
                uint32_t coeff_end = std::min(coeff_tile + tile, dimension);

                for (uint32_t i = ct_tile; i < ct_end; i++) {
                    const NativeVector& a = cts[i]->GetA();
                    for (uint32_t j = coeff_tile; j < coeff_end; j++) {
                        // SoA layout: coeffs_a[j * batch_size + i]
                        impl_->coeffs_a[impl_->SoAIndex(i, j)] = a[j].ConvertToInt<uint64_t>();
                    }
                }
            }
        }
    } else {
        // AoS layout: store sequentially per ciphertext
        #pragma omp parallel for if(batch_size > 100)
        for (uint32_t i = 0; i < batch_size; i++) {
            const NativeVector& a = cts[i]->GetA();
            size_t base = static_cast<size_t>(i) * dimension;
            for (uint32_t j = 0; j < dimension; j++) {
                impl_->coeffs_a[base + j] = a[j].ConvertToInt<uint64_t>();
            }
        }
    }

    // Import 'b' components (simple linear layout)
    for (uint32_t i = 0; i < batch_size; i++) {
        impl_->coeffs_b[i] = cts[i]->GetB().ConvertToInt<uint64_t>();
    }

    impl_->valid = true;
}

void BatchGPULayout::ExportCiphertexts(std::vector<LWECiphertext>& cts) const {
    if (!impl_->valid) {
        cts.clear();
        return;
    }

    uint32_t batch_size = impl_->params.batch_size;
    uint32_t dimension = impl_->params.dimension;
    NativeInteger q = impl_->modulus;

    cts.resize(batch_size);

    if (impl_->params.layout == MemoryLayout::SOA) {
        // Convert SoA back to AoS
        #pragma omp parallel for if(batch_size > 100)
        for (uint32_t i = 0; i < batch_size; i++) {
            NativeVector a(dimension, q);
            for (uint32_t j = 0; j < dimension; j++) {
                a[j] = NativeInteger(impl_->coeffs_a[impl_->SoAIndex(i, j)]);
            }
            NativeInteger b(impl_->coeffs_b[i]);
            cts[i] = std::make_shared<LWECiphertextImpl>(std::move(a), b);
        }
    } else {
        // AoS: direct copy
        #pragma omp parallel for if(batch_size > 100)
        for (uint32_t i = 0; i < batch_size; i++) {
            NativeVector a(dimension, q);
            size_t base = static_cast<size_t>(i) * dimension;
            for (uint32_t j = 0; j < dimension; j++) {
                a[j] = NativeInteger(impl_->coeffs_a[base + j]);
            }
            NativeInteger b(impl_->coeffs_b[i]);
            cts[i] = std::make_shared<LWECiphertextImpl>(std::move(a), b);
        }
    }
}

void BatchGPULayout::ImportPartials(const BatchPartialDecryption& partials) {
    uint32_t batch_size = static_cast<uint32_t>(partials.values.size());

    if (impl_->params.batch_size != batch_size) {
        impl_->params.batch_size = batch_size;
        impl_->params.dimension = 0;  // No 'a' coefficients for partials
        impl_->coeffs_a.clear();
        impl_->coeffs_b.resize(batch_size);
    }

    for (uint32_t i = 0; i < batch_size; i++) {
        impl_->coeffs_b[i] = partials.values[i].ConvertToInt<uint64_t>();
    }

    impl_->valid = true;
}

void BatchGPULayout::ExportPartials(BatchPartialDecryption& partials) const {
    if (!impl_->valid) {
        partials.values.clear();
        return;
    }

    uint32_t batch_size = impl_->params.batch_size;
    partials.values.resize(batch_size);

    for (uint32_t i = 0; i < batch_size; i++) {
        partials.values[i] = NativeInteger(impl_->coeffs_b[i]);
    }
}

// ============================================================================
// Coefficient Access
// ============================================================================

NativeInteger BatchGPULayout::GetA(uint32_t ct_index, uint32_t coeff_index) const {
    if (!impl_->valid || ct_index >= impl_->params.batch_size ||
        coeff_index >= impl_->params.dimension) {
        throw std::out_of_range("Index out of range");
    }

    size_t idx = (impl_->params.layout == MemoryLayout::SOA)
        ? impl_->SoAIndex(ct_index, coeff_index)
        : static_cast<size_t>(ct_index) * impl_->params.dimension + coeff_index;

    return NativeInteger(impl_->coeffs_a[idx]);
}

void BatchGPULayout::SetA(uint32_t ct_index, uint32_t coeff_index, const NativeInteger& val) {
    if (!impl_->valid || ct_index >= impl_->params.batch_size ||
        coeff_index >= impl_->params.dimension) {
        throw std::out_of_range("Index out of range");
    }

    size_t idx = (impl_->params.layout == MemoryLayout::SOA)
        ? impl_->SoAIndex(ct_index, coeff_index)
        : static_cast<size_t>(ct_index) * impl_->params.dimension + coeff_index;

    impl_->coeffs_a[idx] = val.ConvertToInt<uint64_t>();
}

NativeInteger BatchGPULayout::GetB(uint32_t ct_index) const {
    if (!impl_->valid || ct_index >= impl_->params.batch_size) {
        throw std::out_of_range("Index out of range");
    }
    return NativeInteger(impl_->coeffs_b[ct_index]);
}

void BatchGPULayout::SetB(uint32_t ct_index, const NativeInteger& val) {
    if (!impl_->valid || ct_index >= impl_->params.batch_size) {
        throw std::out_of_range("Index out of range");
    }
    impl_->coeffs_b[ct_index] = val.ConvertToInt<uint64_t>();
}

const uint64_t* BatchGPULayout::GetCoeffSlice(uint32_t coeff_index) const {
    if (!impl_->valid || coeff_index >= impl_->params.dimension) {
        return nullptr;
    }
    if (impl_->params.layout != MemoryLayout::SOA) {
        throw std::runtime_error("GetCoeffSlice requires SOA layout");
    }
    return &impl_->coeffs_a[static_cast<size_t>(coeff_index) * impl_->params.batch_size];
}

uint64_t* BatchGPULayout::GetCoeffSliceMut(uint32_t coeff_index) {
    if (!impl_->valid || coeff_index >= impl_->params.dimension) {
        return nullptr;
    }
    if (impl_->params.layout != MemoryLayout::SOA) {
        throw std::runtime_error("GetCoeffSliceMut requires SOA layout");
    }
    return &impl_->coeffs_a[static_cast<size_t>(coeff_index) * impl_->params.batch_size];
}

const uint64_t* BatchGPULayout::GetBSlice() const {
    if (!impl_->valid) {
        return nullptr;
    }
    return impl_->coeffs_b.data();
}

uint64_t* BatchGPULayout::GetBSliceMut() {
    if (!impl_->valid) {
        return nullptr;
    }
    return impl_->coeffs_b.data();
}

// ============================================================================
// Vectorized Operations
// ============================================================================

void BatchGPULayout::BatchInnerProduct(
    const NativeVector& key_share,
    std::vector<NativeInteger>& results
) const {
    if (!impl_->valid) {
        results.clear();
        return;
    }

    uint32_t batch_size = impl_->params.batch_size;
    uint32_t dimension = impl_->params.dimension;
    uint64_t q = impl_->modulus.ConvertToInt<uint64_t>();

    if (key_share.GetLength() != dimension) {
        throw std::invalid_argument("Key share dimension mismatch");
    }

    results.resize(batch_size);

    // Extract key share to raw buffer for vectorized access
    std::vector<uint64_t> s(dimension);
    for (uint32_t j = 0; j < dimension; j++) {
        s[j] = key_share[j].ConvertToInt<uint64_t>();
    }

    // Compute inner products
    BatchInnerProductRaw(s.data(), dimension, q,
                         reinterpret_cast<uint64_t*>(results.data()));

    // Convert results to NativeInteger
    // (BatchInnerProductRaw stores raw uint64_t, we need to wrap)
    std::vector<uint64_t> raw_results(batch_size);
    BatchInnerProductRaw(s.data(), dimension, q, raw_results.data());

    for (uint32_t i = 0; i < batch_size; i++) {
        results[i] = NativeInteger(raw_results[i]);
    }
}

void BatchGPULayout::BatchInnerProductRaw(
    const uint64_t* key_share_ptr,
    uint32_t key_len,
    uint64_t q,
    uint64_t* results_ptr
) const {
    if (!impl_->valid || key_len != impl_->params.dimension) {
        return;
    }

    uint32_t batch_size = impl_->params.batch_size;
    uint32_t dimension = impl_->params.dimension;

    // Initialize results to zero
    std::memset(results_ptr, 0, batch_size * sizeof(uint64_t));

    if (impl_->params.layout == MemoryLayout::SOA) {
        // OPTIMIZED: SoA layout enables vectorized multiply-accumulate
        //
        // For each coefficient j:
        //   1. Load s[j] once (amortized across batch)
        //   2. Load all a[j,*] in one coalesced access
        //   3. Multiply-add: results[i] += a[j,i] * s[j]
        //
        // This is ideal for GPU: all threads in warp access same s[j],
        // different but adjacent a[j,i].

        for (uint32_t j = 0; j < dimension; j++) {
            uint64_t s_j = key_share_ptr[j];
            const uint64_t* a_slice = GetCoeffSlice(j);

            // Vectorized multiply-accumulate
            // Compiler can auto-vectorize this loop (AVX2/AVX-512)
            #pragma omp simd
            for (uint32_t i = 0; i < batch_size; i++) {
                // Modular multiply-add
                // Using 128-bit intermediate to avoid overflow
                __uint128_t prod = static_cast<__uint128_t>(a_slice[i]) * s_j;
                __uint128_t sum = results_ptr[i] + (prod % q);
                results_ptr[i] = sum % q;
            }
        }
    } else {
        // AoS layout: less efficient but still parallel across batch
        #pragma omp parallel for if(batch_size > 100)
        for (uint32_t i = 0; i < batch_size; i++) {
            uint64_t sum = 0;
            size_t base = static_cast<size_t>(i) * dimension;

            for (uint32_t j = 0; j < dimension; j++) {
                __uint128_t prod = static_cast<__uint128_t>(impl_->coeffs_a[base + j]) * key_share_ptr[j];
                sum = (sum + prod % q) % q;
            }

            results_ptr[i] = sum;
        }
    }
}

void BatchGPULayout::BatchModAdd(
    const BatchGPULayout& other,
    BatchGPULayout& result
) const {
    if (!impl_->valid || !other.impl_->valid) {
        throw std::invalid_argument("Invalid layout");
    }

    if (impl_->params.batch_size != other.impl_->params.batch_size ||
        impl_->params.dimension != other.impl_->params.dimension) {
        throw std::invalid_argument("Dimension mismatch");
    }

    uint32_t batch_size = impl_->params.batch_size;
    uint32_t dimension = impl_->params.dimension;
    uint64_t q = impl_->modulus.ConvertToInt<uint64_t>();

    // Ensure result has correct dimensions
    if (result.impl_->params.batch_size != batch_size ||
        result.impl_->params.dimension != dimension) {
        result.Resize(batch_size, dimension, impl_->modulus);
    }

    size_t total = static_cast<size_t>(batch_size) * dimension;

    #pragma omp parallel for if(total > 10000)
    for (size_t i = 0; i < total; i++) {
        __uint128_t sum = impl_->coeffs_a[i] + other.impl_->coeffs_a[i];
        result.impl_->coeffs_a[i] = sum % q;
    }

    #pragma omp parallel for if(batch_size > 100)
    for (uint32_t i = 0; i < batch_size; i++) {
        __uint128_t sum = impl_->coeffs_b[i] + other.impl_->coeffs_b[i];
        result.impl_->coeffs_b[i] = sum % q;
    }

    result.impl_->valid = true;
}

void BatchGPULayout::BatchScalarMul(
    const NativeInteger& scalar,
    BatchGPULayout& result
) const {
    if (!impl_->valid) {
        throw std::invalid_argument("Invalid layout");
    }

    uint32_t batch_size = impl_->params.batch_size;
    uint32_t dimension = impl_->params.dimension;
    uint64_t q = impl_->modulus.ConvertToInt<uint64_t>();
    uint64_t s = scalar.ConvertToInt<uint64_t>();

    if (result.impl_->params.batch_size != batch_size ||
        result.impl_->params.dimension != dimension) {
        result.Resize(batch_size, dimension, impl_->modulus);
    }

    size_t total = static_cast<size_t>(batch_size) * dimension;

    #pragma omp parallel for if(total > 10000)
    for (size_t i = 0; i < total; i++) {
        __uint128_t prod = static_cast<__uint128_t>(impl_->coeffs_a[i]) * s;
        result.impl_->coeffs_a[i] = prod % q;
    }

    #pragma omp parallel for if(batch_size > 100)
    for (uint32_t i = 0; i < batch_size; i++) {
        __uint128_t prod = static_cast<__uint128_t>(impl_->coeffs_b[i]) * s;
        result.impl_->coeffs_b[i] = prod % q;
    }

    result.impl_->valid = true;
}

// ============================================================================
// Layout Properties
// ============================================================================

uint32_t BatchGPULayout::BatchSize() const {
    return impl_->params.batch_size;
}

uint32_t BatchGPULayout::Dimension() const {
    return impl_->params.dimension;
}

NativeInteger BatchGPULayout::Modulus() const {
    return impl_->modulus;
}

MemoryLayout BatchGPULayout::Layout() const {
    return impl_->params.layout;
}

const LayoutParams& BatchGPULayout::Params() const {
    return impl_->params;
}

size_t BatchGPULayout::MemoryBytes() const {
    return impl_->coeffs_a.size() * sizeof(uint64_t) +
           impl_->coeffs_b.size() * sizeof(uint64_t);
}

bool BatchGPULayout::IsValid() const {
    return impl_->valid;
}

void BatchGPULayout::Clear() {
    impl_->Clear();
}

void BatchGPULayout::Resize(uint32_t batch_size, uint32_t dimension, const NativeInteger& q) {
    impl_->params.batch_size = batch_size;
    impl_->params.dimension = dimension;
    impl_->modulus = q;
    impl_->Allocate();
    impl_->valid = false;  // Data not yet populated
}

const uint64_t* BatchGPULayout::RawCoeffsA() const {
    return impl_->coeffs_a.data();
}

uint64_t* BatchGPULayout::RawCoeffsAMut() {
    return impl_->coeffs_a.data();
}

// ============================================================================
// Transpose Operations
// ============================================================================

void TransposeAoSToSoA(
    const uint64_t* src,
    uint64_t* dst,
    uint32_t batch_size,
    uint32_t dimension
) {
    // Cache-efficient tiled transpose
    //
    // Process TILE_DIM x TILE_DIM blocks to maximize cache reuse.
    // Each tile is small enough to fit in L1 cache.

    const uint32_t tile = gpu::TILE_DIM;

    #pragma omp parallel for collapse(2) if(batch_size * dimension > 10000)
    for (uint32_t ct_tile = 0; ct_tile < batch_size; ct_tile += tile) {
        for (uint32_t coeff_tile = 0; coeff_tile < dimension; coeff_tile += tile) {
            uint32_t ct_end = std::min(ct_tile + tile, batch_size);
            uint32_t coeff_end = std::min(coeff_tile + tile, dimension);

            for (uint32_t i = ct_tile; i < ct_end; i++) {
                for (uint32_t j = coeff_tile; j < coeff_end; j++) {
                    // AoS: src[i * dimension + j]
                    // SoA: dst[j * batch_size + i]
                    dst[static_cast<size_t>(j) * batch_size + i] =
                        src[static_cast<size_t>(i) * dimension + j];
                }
            }
        }
    }
}

void TransposeSoAToAoS(
    const uint64_t* src,
    uint64_t* dst,
    uint32_t batch_size,
    uint32_t dimension
) {
    const uint32_t tile = gpu::TILE_DIM;

    #pragma omp parallel for collapse(2) if(batch_size * dimension > 10000)
    for (uint32_t ct_tile = 0; ct_tile < batch_size; ct_tile += tile) {
        for (uint32_t coeff_tile = 0; coeff_tile < dimension; coeff_tile += tile) {
            uint32_t ct_end = std::min(ct_tile + tile, batch_size);
            uint32_t coeff_end = std::min(coeff_tile + tile, dimension);

            for (uint32_t i = ct_tile; i < ct_end; i++) {
                for (uint32_t j = coeff_tile; j < coeff_end; j++) {
                    // SoA: src[j * batch_size + i]
                    // AoS: dst[i * dimension + j]
                    dst[static_cast<size_t>(i) * dimension + j] =
                        src[static_cast<size_t>(j) * batch_size + i];
                }
            }
        }
    }
}

void TransposeInPlace(
    uint64_t* data,
    uint32_t batch_size,
    uint32_t dimension,
    bool to_soa
) {
    // In-place transpose using auxiliary buffer
    // Could be optimized with cycle-following for square matrices

    size_t total = static_cast<size_t>(batch_size) * dimension;
    std::vector<uint64_t> temp(total);

    if (to_soa) {
        TransposeAoSToSoA(data, temp.data(), batch_size, dimension);
    } else {
        TransposeSoAToAoS(data, temp.data(), batch_size, dimension);
    }

    std::memcpy(data, temp.data(), total * sizeof(uint64_t));
}

// ============================================================================
// GPU Integration Helpers
// ============================================================================

uint32_t ComputeOptimalBatchSize(uint32_t dimension, size_t available_memory) {
    // Memory per ciphertext: dimension * 8 bytes (coeffs) + 8 bytes (b)
    size_t bytes_per_ct = dimension * sizeof(uint64_t) + sizeof(uint64_t);

    // Also need space for key share and results
    size_t overhead = dimension * sizeof(uint64_t) * 2;

    if (available_memory <= overhead) {
        return gpu::WARP_SIZE;  // Minimum batch
    }

    size_t usable = available_memory - overhead;
    uint32_t max_batch = static_cast<uint32_t>(usable / bytes_per_ct);

    // Cap at maximum and round to warp multiple
    max_batch = std::min(max_batch, gpu::MAX_BATCH_SIZE);
    max_batch = RoundToWarp(max_batch);

    // Ensure at least one warp
    return std::max(max_batch, gpu::WARP_SIZE);
}

std::vector<uint64_t> PrepareKeyShareForGPU(const NativeVector& key_share) {
    uint32_t n = key_share.GetLength();

    // Pad to cache line boundary for efficient broadcast
    uint32_t padded = ((n * sizeof(uint64_t) + gpu::CACHE_LINE_BYTES - 1)
                       / gpu::CACHE_LINE_BYTES) * gpu::CACHE_LINE_BYTES / sizeof(uint64_t);

    std::vector<uint64_t> result(padded, 0);

    for (uint32_t i = 0; i < n; i++) {
        result[i] = key_share[i].ConvertToInt<uint64_t>();
    }

    return result;
}

// ============================================================================
// BatchGPUDecrypt Implementation
// ============================================================================

struct BatchGPUDecrypt::Impl {
    BinFHEContext* cc = nullptr;
    KeyShare key_share;
    std::vector<uint64_t> prepared_key;
    NativeInteger q;
    uint32_t dimension = 0;
    Timings last_timings{};

    void Initialize(BinFHEContext& context, const KeyShare& ks) {
        cc = &context;
        key_share = ks;

        auto params = cc->GetParams()->GetLWEParams();
        q = params->Getq();
        dimension = params->Getn();

        prepared_key = PrepareKeyShareForGPU(key_share.share);
    }
};

BatchGPUDecrypt::BatchGPUDecrypt() : impl_(std::make_unique<Impl>()) {}
BatchGPUDecrypt::~BatchGPUDecrypt() = default;

void BatchGPUDecrypt::Initialize(BinFHEContext& cc, const KeyShare& key_share) {
    impl_->Initialize(cc, key_share);
}

ThresholdBatchResult BatchGPUDecrypt::Process(
    const std::vector<LWECiphertext>& cts,
    BatchPartialDecryption& out
) {
    auto start_total = std::chrono::high_resolution_clock::now();
    ThresholdBatchResult result = ThresholdBatchResult::Success(cts.size());

    if (cts.empty()) {
        return result;
    }

    try {
        // Step 1: Import and transpose to SoA layout
        auto start_transpose = std::chrono::high_resolution_clock::now();
        BatchGPULayout layout(cts, MemoryLayout::SOA);
        auto end_transpose = std::chrono::high_resolution_clock::now();
        impl_->last_timings.transpose_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_transpose - start_transpose).count();

        // Step 2: Compute inner products
        result = ProcessLayout(layout, out);

    } catch (const std::exception& e) {
        result = ThresholdBatchResult::Failure(e.what(), cts.size());
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    impl_->last_timings.total_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_total - start_total).count();
    result.time_total_ns = impl_->last_timings.total_ns;

    return result;
}

ThresholdBatchResult BatchGPUDecrypt::ProcessLayout(
    const BatchGPULayout& layout,
    BatchPartialDecryption& out
) {
    auto start = std::chrono::high_resolution_clock::now();
    (void)start;  // TODO: Used for timing in production builds
    ThresholdBatchResult result = ThresholdBatchResult::Success(layout.BatchSize());

    if (!layout.IsValid()) {
        return ThresholdBatchResult::Failure("Invalid layout", 0);
    }

    try {
        uint32_t batch_size = layout.BatchSize();
        uint32_t dimension = layout.Dimension();
        uint64_t q_val = impl_->q.ConvertToInt<uint64_t>();

        // Compute batched inner products
        std::vector<uint64_t> raw_results(batch_size);

        auto start_compute = std::chrono::high_resolution_clock::now();
        layout.BatchInnerProductRaw(
            impl_->prepared_key.data(),
            dimension,
            q_val,
            raw_results.data()
        );
        auto end_compute = std::chrono::high_resolution_clock::now();
        impl_->last_timings.compute_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_compute - start_compute).count();

        // Export results
        auto start_export = std::chrono::high_resolution_clock::now();
        out.party_id = impl_->key_share.party_id;
        out.values.resize(batch_size);
        for (uint32_t i = 0; i < batch_size; i++) {
            out.values[i] = NativeInteger(raw_results[i]);
        }
        auto end_export = std::chrono::high_resolution_clock::now();
        impl_->last_timings.export_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_export - start_export).count();

        result.time_compute_ns = impl_->last_timings.compute_ns;
        result.processed = batch_size;

    } catch (const std::exception& e) {
        result = ThresholdBatchResult::Failure(e.what(), layout.BatchSize());
    }

    return result;
}

BatchGPUDecrypt::Timings BatchGPUDecrypt::LastTimings() const {
    return impl_->last_timings;
}

// ============================================================================
// Utilities
// ============================================================================

bool VerifyLayoutRoundTrip(const std::vector<LWECiphertext>& cts) {
    if (cts.empty()) {
        return true;
    }

    // Import to SoA
    BatchGPULayout layout(cts, MemoryLayout::SOA);

    // Export back
    std::vector<LWECiphertext> exported;
    layout.ExportCiphertexts(exported);

    // Verify equality
    if (exported.size() != cts.size()) {
        return false;
    }

    for (size_t i = 0; i < cts.size(); i++) {
        if (*cts[i] != *exported[i]) {
            return false;
        }
    }

    return true;
}

uint64_t BenchmarkTranspose(
    uint32_t batch_size,
    uint32_t dimension,
    uint32_t iterations
) {
    size_t total = static_cast<size_t>(batch_size) * dimension;
    std::vector<uint64_t> src(total);
    std::vector<uint64_t> dst(total);

    // Initialize with test data
    for (size_t i = 0; i < total; i++) {
        src[i] = i;
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (uint32_t iter = 0; iter < iterations; iter++) {
        TransposeAoSToSoA(src.data(), dst.data(), batch_size, dimension);
    }

    auto end = std::chrono::high_resolution_clock::now();
    uint64_t total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    return total_ns / iterations;
}

} // namespace threshold
} // namespace lbcrypto
