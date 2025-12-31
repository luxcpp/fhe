// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// GPU-Optimized Memory Layout for Batched Threshold FHE
//
// INNOVATION: Structure-of-Arrays (SoA) layout optimized for GPU coalesced access
//
// Traditional Array-of-Structures (AoS):
//   Memory: [ct0[0..N], ct1[0..N], ct2[0..N], ...]
//   Problem: Inner product <a_i, s> has strided access to s across ciphertexts
//   GPU threads accessing coefficient j of different cts hit different cache lines
//
// Optimized Structure-of-Arrays (SoA):
//   Memory: [ct0[0],ct1[0],ct2[0],...], [ct0[1],ct1[1],...], ...
//   Benefit: All threads in warp access coefficient j together - coalesced!
//   Key share s[j] loaded once, used by all batch elements
//
// Performance model for batch of B ciphertexts, dimension N:
//   AoS inner product: B * N memory transactions (strided)
//   SoA inner product: N memory transactions (coalesced) + B element parallelism
//
// This layout enables:
// 1. Coalesced memory access for GPU warps
// 2. Amortized key share loading (load s[j] once, use for all B ciphertexts)
// 3. Efficient transpose via shared memory tiling
// 4. Vectorized multiply-accumulate across batch dimension

#ifndef THRESHOLD_BATCH_GPU_LAYOUT_H
#define THRESHOLD_BATCH_GPU_LAYOUT_H

#include "lwe-ciphertext.h"
#include "threshold/batch_threshold.h"
#include "math/math-hal.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace lbcrypto {
namespace threshold {

// ============================================================================
// Configuration Constants
// ============================================================================

namespace gpu {

// GPU warp size (NVIDIA: 32, AMD: 64)
constexpr uint32_t WARP_SIZE = 32;

// Tile size for shared memory transpose
// Must be multiple of warp size for coalesced access
constexpr uint32_t TILE_DIM = 32;

// Maximum batch size for single kernel launch
// Balances parallelism vs. register pressure
constexpr uint32_t MAX_BATCH_SIZE = 16384;

// Cache line size in bytes (typical: 128 bytes)
constexpr uint32_t CACHE_LINE_BYTES = 128;

// Alignment for coefficient storage (8 bytes for uint64_t)
constexpr uint32_t COEFF_ALIGN = 8;

} // namespace gpu

// ============================================================================
// Memory Layout Descriptors
// ============================================================================

/**
 * @brief Memory layout for batch coefficient storage
 */
enum class MemoryLayout : uint8_t {
    AOS,    // Array of Structures: [ct0[0..N], ct1[0..N], ...]
    SOA,    // Structure of Arrays: [all coeff 0], [all coeff 1], ...
    HYBRID  // Tiled: groups of coefficients for cache efficiency
};

/**
 * @brief Layout parameters for GPU memory allocation
 */
struct LayoutParams {
    uint32_t batch_size;      // Number of ciphertexts
    uint32_t dimension;       // LWE dimension N
    uint32_t modulus_bits;    // Bits in modulus q
    MemoryLayout layout;      // Memory organization
    uint32_t tile_size;       // Tile dimension for hybrid layout

    // Computed values
    size_t total_coeffs() const { return static_cast<size_t>(batch_size) * dimension; }
    size_t storage_bytes() const { return total_coeffs() * sizeof(uint64_t); }
    size_t padded_batch() const;   // Batch size padded to warp multiple
    size_t padded_dim() const;     // Dimension padded to tile multiple
};

// ============================================================================
// BatchGPULayout - Core SoA Storage Class
// ============================================================================

/**
 * @brief GPU-optimized batch storage for LWE ciphertext coefficients
 *
 * Stores coefficients in Structure-of-Arrays layout:
 *   coeffs_a[j * batch_size + i] = ciphertext i, coefficient j
 *   coeffs_b[i] = ciphertext i, 'b' component
 *
 * This enables coalesced GPU memory access:
 * - Thread i processes ciphertext i
 * - All threads access coefficient j together (coalesced read)
 * - Key share s[j] loaded once per warp, broadcast to all threads
 *
 * Memory layout example (batch=4, dim=3):
 *   AoS: [a0[0],a0[1],a0[2]], [a1[0],a1[1],a1[2]], [a2[0],a2[1],a2[2]], [a3[0],a3[1],a3[2]]
 *   SoA: [a0[0],a1[0],a2[0],a3[0]], [a0[1],a1[1],a2[1],a3[1]], [a0[2],a1[2],a2[2],a3[2]]
 */
class BatchGPULayout {
public:
    /**
     * @brief Create empty layout
     */
    BatchGPULayout();

    /**
     * @brief Create layout with specified parameters
     */
    explicit BatchGPULayout(const LayoutParams& params);

    /**
     * @brief Create from vector of ciphertexts (performs AoS->SoA transpose)
     */
    BatchGPULayout(const std::vector<LWECiphertext>& cts, MemoryLayout layout = MemoryLayout::SOA);

    ~BatchGPULayout();

    // Move-only (large memory buffers)
    BatchGPULayout(BatchGPULayout&&) noexcept;
    BatchGPULayout& operator=(BatchGPULayout&&) noexcept;
    BatchGPULayout(const BatchGPULayout&) = delete;
    BatchGPULayout& operator=(const BatchGPULayout&) = delete;

    // ========================================================================
    // Data Import/Export
    // ========================================================================

    /**
     * @brief Import ciphertexts, converting to SoA layout
     *
     * OPTIMIZATION: Uses tiled transpose via cache-efficient blocking
     *
     * @param cts Input ciphertexts in AoS format
     */
    void ImportCiphertexts(const std::vector<LWECiphertext>& cts);

    /**
     * @brief Export back to ciphertext vector (SoA->AoS transpose)
     *
     * @param cts Output ciphertexts
     */
    void ExportCiphertexts(std::vector<LWECiphertext>& cts) const;

    /**
     * @brief Import partial decryption values (already in batch format)
     *
     * @param partials BatchPartialDecryption containing values for each ct
     */
    void ImportPartials(const BatchPartialDecryption& partials);

    /**
     * @brief Export partial decryption values
     *
     * @param partials Output partial decryptions
     */
    void ExportPartials(BatchPartialDecryption& partials) const;

    // ========================================================================
    // SoA Coefficient Access
    // ========================================================================

    /**
     * @brief Get coefficient j of ciphertext i
     *
     * SoA index: coeffs_a[j * batch_size + i]
     */
    NativeInteger GetA(uint32_t ct_index, uint32_t coeff_index) const;

    /**
     * @brief Set coefficient j of ciphertext i
     */
    void SetA(uint32_t ct_index, uint32_t coeff_index, const NativeInteger& val);

    /**
     * @brief Get 'b' component of ciphertext i
     */
    NativeInteger GetB(uint32_t ct_index) const;

    /**
     * @brief Set 'b' component of ciphertext i
     */
    void SetB(uint32_t ct_index, const NativeInteger& val);

    /**
     * @brief Get pointer to all coefficients at position j (for vectorized ops)
     *
     * Returns pointer to coeffs_a[j * batch_size], length = batch_size
     * Enables vectorized operations across entire batch for coefficient j
     */
    const uint64_t* GetCoeffSlice(uint32_t coeff_index) const;
    uint64_t* GetCoeffSliceMut(uint32_t coeff_index);

    /**
     * @brief Get pointer to all 'b' components
     */
    const uint64_t* GetBSlice() const;
    uint64_t* GetBSliceMut();

    // ========================================================================
    // Vectorized Operations (GPU-optimized)
    // ========================================================================

    /**
     * @brief Batched inner product with key share
     *
     * Computes <a_i, s> for all ciphertexts i in parallel.
     *
     * INNOVATION: SoA layout enables coalesced access pattern:
     *   for j in 0..N:
     *       s_j = s[j]                    // Load once
     *       for i in 0..B (parallel):
     *           result[i] += a[j,i] * s_j // Coalesced read of a[j,*]
     *
     * @param key_share Secret key share s
     * @param results Output: <a_i, s> for each ciphertext
     */
    void BatchInnerProduct(
        const NativeVector& key_share,
        std::vector<NativeInteger>& results
    ) const;

    /**
     * @brief Batched inner product with raw pointer (for GPU integration)
     *
     * @param key_share_ptr Pointer to key share coefficients
     * @param key_len Length of key share
     * @param q Modulus
     * @param results_ptr Output pointer (must have batch_size elements)
     */
    void BatchInnerProductRaw(
        const uint64_t* key_share_ptr,
        uint32_t key_len,
        uint64_t q,
        uint64_t* results_ptr
    ) const;

    /**
     * @brief Batched modular addition: results[i] = a[i] + b[i] mod q
     *
     * @param other Other BatchGPULayout (must have same dimensions)
     * @param results Output layout
     */
    void BatchModAdd(
        const BatchGPULayout& other,
        BatchGPULayout& results
    ) const;

    /**
     * @brief Batched scalar multiply: results[i] = scalar * a[i] mod q
     *
     * @param scalar Scalar multiplier
     * @param results Output layout
     */
    void BatchScalarMul(
        const NativeInteger& scalar,
        BatchGPULayout& results
    ) const;

    // ========================================================================
    // Layout Properties
    // ========================================================================

    uint32_t BatchSize() const;
    uint32_t Dimension() const;
    NativeInteger Modulus() const;
    MemoryLayout Layout() const;
    const LayoutParams& Params() const;

    /**
     * @brief Total memory footprint in bytes
     */
    size_t MemoryBytes() const;

    /**
     * @brief Check if layout is valid and contains data
     */
    bool IsValid() const;

    // ========================================================================
    // Memory Management
    // ========================================================================

    /**
     * @brief Release memory and reset to empty state
     */
    void Clear();

    /**
     * @brief Resize layout (reallocates if necessary)
     */
    void Resize(uint32_t batch_size, uint32_t dimension, const NativeInteger& q);

    /**
     * @brief Get raw coefficient buffer (for GPU memcpy)
     */
    const uint64_t* RawCoeffsA() const;
    uint64_t* RawCoeffsAMut();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// Transpose Operations
// ============================================================================

/**
 * @brief Efficient AoS to SoA transpose
 *
 * Uses cache-efficient tiled transpose:
 * 1. Load TILE_DIM x TILE_DIM block into shared memory
 * 2. Transpose within shared memory (no bank conflicts)
 * 3. Write transposed block to output
 *
 * @param src Source coefficients in AoS layout
 * @param dst Destination coefficients in SoA layout
 * @param batch_size Number of ciphertexts
 * @param dimension Coefficient count per ciphertext
 */
void TransposeAoSToSoA(
    const uint64_t* src,
    uint64_t* dst,
    uint32_t batch_size,
    uint32_t dimension
);

/**
 * @brief Efficient SoA to AoS transpose
 *
 * @param src Source coefficients in SoA layout
 * @param dst Destination coefficients in AoS layout
 * @param batch_size Number of ciphertexts
 * @param dimension Coefficient count per ciphertext
 */
void TransposeSoAToAoS(
    const uint64_t* src,
    uint64_t* dst,
    uint32_t batch_size,
    uint32_t dimension
);

/**
 * @brief In-place transpose using auxiliary buffer
 *
 * For when separate src/dst is not available.
 *
 * @param data Coefficient data
 * @param batch_size Number of ciphertexts
 * @param dimension Coefficient count per ciphertext
 * @param to_soa True for AoS->SoA, false for SoA->AoS
 */
void TransposeInPlace(
    uint64_t* data,
    uint32_t batch_size,
    uint32_t dimension,
    bool to_soa
);

// ============================================================================
// GPU Integration Helpers
// ============================================================================

/**
 * @brief Compute optimal batch size for GPU execution
 *
 * Balances:
 * - Occupancy (enough threads to hide latency)
 * - Register pressure (per-thread state)
 * - Shared memory (transpose tiles)
 *
 * @param dimension LWE dimension
 * @param available_memory Available GPU memory in bytes
 * @return Optimal batch size
 */
uint32_t ComputeOptimalBatchSize(
    uint32_t dimension,
    size_t available_memory
);

/**
 * @brief Memory layout for key share in GPU constant memory
 *
 * Key shares are accessed uniformly by all threads.
 * Storing in constant memory enables broadcast to entire warp.
 *
 * @param key_share Key share vector
 * @return Aligned buffer suitable for GPU constant memory
 */
std::vector<uint64_t> PrepareKeyShareForGPU(const NativeVector& key_share);

/**
 * @brief Batch size rounded up to warp multiple
 *
 * Ensures full warp utilization for coalesced access.
 */
inline uint32_t RoundToWarp(uint32_t batch_size) {
    return ((batch_size + gpu::WARP_SIZE - 1) / gpu::WARP_SIZE) * gpu::WARP_SIZE;
}

/**
 * @brief Dimension rounded up to tile multiple
 *
 * Ensures efficient transpose tiling.
 */
inline uint32_t RoundToTile(uint32_t dimension) {
    return ((dimension + gpu::TILE_DIM - 1) / gpu::TILE_DIM) * gpu::TILE_DIM;
}

// ============================================================================
// BatchGPUDecrypt - Integrated GPU Partial Decryption
// ============================================================================

/**
 * @brief GPU-optimized batch partial decryption
 *
 * Combines BatchGPULayout with BatchPartialDecrypt for maximum throughput.
 *
 * Workflow:
 * 1. Import ciphertexts to SoA layout (transpose)
 * 2. Load key share to GPU constant memory
 * 3. Launch batched inner product kernel
 * 4. Export partial decryptions
 *
 * Performance:
 *   Traditional: B * N memory transactions
 *   GPU SoA: N coalesced transactions + B parallelism
 */
class BatchGPUDecrypt {
public:
    BatchGPUDecrypt();
    ~BatchGPUDecrypt();

    // Non-copyable
    BatchGPUDecrypt(const BatchGPUDecrypt&) = delete;
    BatchGPUDecrypt& operator=(const BatchGPUDecrypt&) = delete;

    /**
     * @brief Initialize with context and key share
     *
     * @param cc BinFHE context
     * @param key_share This party's key share
     */
    void Initialize(BinFHEContext& cc, const KeyShare& key_share);

    /**
     * @brief Process batch of ciphertexts
     *
     * @param cts Input ciphertexts
     * @param out Output partial decryptions
     * @return Result with timing breakdown
     */
    ThresholdBatchResult Process(
        const std::vector<LWECiphertext>& cts,
        BatchPartialDecryption& out
    );

    /**
     * @brief Process with pre-transposed layout
     *
     * Use when layout is already in SoA format.
     *
     * @param layout Pre-transposed ciphertext layout
     * @param out Output partial decryptions
     * @return Result with timing breakdown
     */
    ThresholdBatchResult ProcessLayout(
        const BatchGPULayout& layout,
        BatchPartialDecryption& out
    );

    /**
     * @brief Get timing statistics
     */
    struct Timings {
        uint64_t transpose_ns;    // AoS->SoA transpose
        uint64_t compute_ns;      // Inner product computation
        uint64_t export_ns;       // Result export
        uint64_t total_ns;
    };
    Timings LastTimings() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// Utilities
// ============================================================================

/**
 * @brief Verify layout correctness by round-trip test
 *
 * Import -> Export should yield identical ciphertexts.
 *
 * @param cts Test ciphertexts
 * @return True if round-trip preserves data
 */
bool VerifyLayoutRoundTrip(const std::vector<LWECiphertext>& cts);

/**
 * @brief Benchmark transpose performance
 *
 * @param batch_size Number of ciphertexts
 * @param dimension Coefficient dimension
 * @param iterations Number of iterations
 * @return Average nanoseconds per transpose
 */
uint64_t BenchmarkTranspose(
    uint32_t batch_size,
    uint32_t dimension,
    uint32_t iterations = 100
);

} // namespace threshold
} // namespace lbcrypto

#endif // THRESHOLD_BATCH_GPU_LAYOUT_H
