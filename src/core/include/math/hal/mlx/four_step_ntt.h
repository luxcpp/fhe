//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2024-2025, Lux Industries Inc
//
// All rights reserved.
//
// Four-Step NTT Optimized for Apple Metal Threadgroup Memory and SIMDgroup
//
// This implements the Four-Step NTT algorithm specifically tuned for:
// - 32KB Metal threadgroup memory (vs 48KB CUDA shared memory)
// - 32-lane SIMDgroup operations (simd_shuffle)
// - Coalesced memory access patterns for unified memory
// - Integer-only arithmetic for FHE determinism
//
// See patent: PAT-FHE-010-four-step-ntt-metal.md
//==================================================================================

#ifndef LBCRYPTO_MATH_HAL_MLX_FOUR_STEP_NTT_H
#define LBCRYPTO_MATH_HAL_MLX_FOUR_STEP_NTT_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>

#ifdef WITH_METAL
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#endif

namespace lbcrypto {
namespace metal_backend {

//==================================================================================
// Constants and Tile Configuration
//==================================================================================

/**
 * @brief Metal-specific constraints
 */
constexpr uint32_t METAL_THREADGROUP_MEMORY_SIZE = 32768;  // 32KB
constexpr uint32_t METAL_SIMDGROUP_SIZE = 32;              // 32 threads per SIMD
constexpr uint32_t METAL_MAX_THREADS_PER_THREADGROUP = 1024;

/**
 * @brief Tile sizes for different N values
 *
 * For 64-bit elements, we can fit TILE_SIZE = 32KB / 8 = 4096 elements.
 * We add padding to avoid bank conflicts, so actual usable is slightly less.
 */
struct TileConfig {
    uint32_t N1;          // Column dimension
    uint32_t N2;          // Row dimension
    uint32_t tile_stride; // Padded stride for bank conflict avoidance
    uint32_t num_tiles;   // Number of tiles needed

    static TileConfig ForN(uint32_t N) {
        TileConfig cfg;

        // Select tile dimensions based on N
        if (N <= 1024) {
            // Small N: single tile, process entirely in threadgroup memory
            cfg.N1 = N;
            cfg.N2 = 1;
            cfg.tile_stride = N;
            cfg.num_tiles = 1;
        } else if (N <= 4096) {
            // Medium N: 64x64 tile (4096 elements = 32KB exactly)
            cfg.N1 = 64;
            cfg.N2 = 64;
            cfg.tile_stride = 65;  // Pad by 1 for bank conflict avoidance
            cfg.num_tiles = N / 4096;
        } else if (N <= 16384) {
            // Large N: 128x128 with 4 tiles
            cfg.N1 = 128;
            cfg.N2 = 128;
            cfg.tile_stride = 129;
            cfg.num_tiles = N / 16384 * 4;  // 4 tiles per 16K
        } else {
            // Very large N: 256x256 (process in chunks)
            cfg.N1 = 256;
            cfg.N2 = 256;
            cfg.tile_stride = 257;
            cfg.num_tiles = N / 65536 * 4;
        }

        return cfg;
    }
};

/**
 * @brief Parameters passed to Metal kernels
 */
struct FourStepNTTParams {
    uint64_t Q;             // Prime modulus
    uint64_t mu;            // Barrett constant: floor(2^64 / Q)
    uint64_t N_inv;         // N^{-1} mod Q for inverse NTT
    uint64_t N_inv_precon;  // Barrett precomputation for N_inv
    uint32_t N;             // Total ring dimension
    uint32_t N1;            // Column dimension for Four-Step
    uint32_t N2;            // Row dimension for Four-Step
    uint32_t log_N1;        // log2(N1)
    uint32_t log_N2;        // log2(N2)
    uint32_t tile_stride;   // Padded stride
    uint32_t batch_size;    // Number of polynomials to process
};

//==================================================================================
// FourStepNTT Class - Main Interface
//==================================================================================

#ifdef WITH_METAL

/**
 * @brief Four-Step NTT implementation optimized for Apple Metal
 *
 * The Four-Step algorithm decomposes NTT(N) into:
 *   1. N2 parallel column NTTs of size N1
 *   2. Twiddle factor multiplication by omega^(i*j)
 *   3. Matrix transpose (N1 x N2 -> N2 x N1)
 *   4. N1 parallel row NTTs of size N2
 *
 * This exploits Metal's threadgroup memory and SIMDgroup operations for:
 *   - In-memory column NTTs (fit tile in 32KB threadgroup memory)
 *   - Radix-32 butterflies using simd_shuffle (zero sync overhead)
 *   - Bank-conflict-free transpose with padding
 *   - Fused twiddle-transpose to minimize memory traffic
 *
 * Usage:
 *   FourStepNTT ntt(16384, prime_modulus, config);
 *   ntt.Forward(input, output);
 *   ntt.Inverse(output, result);
 */
class FourStepNTT {
public:
    /**
     * @brief Construct FourStepNTT with given parameters
     *
     * @param n Ring dimension (must be power of 2, >= 1024)
     * @param q Prime modulus (must be < 2^62 for Barrett reduction)
     * @param device Metal device (nullptr = default device)
     */
    FourStepNTT(uint32_t n, uint64_t q, id<MTLDevice> device = nullptr);

    ~FourStepNTT();

    // Disable copy (Metal resources are not copyable)
    FourStepNTT(const FourStepNTT&) = delete;
    FourStepNTT& operator=(const FourStepNTT&) = delete;

    // Allow move
    FourStepNTT(FourStepNTT&&) noexcept;
    FourStepNTT& operator=(FourStepNTT&&) noexcept;

    //--------------------------------------------------------------------------
    // Core NTT Operations
    //--------------------------------------------------------------------------

    /**
     * @brief Forward NTT (time domain -> frequency domain)
     *
     * @param input Input polynomial coefficients [N]
     * @param output Output NTT coefficients [N] (can alias input for in-place)
     */
    void Forward(const std::vector<uint64_t>& input, std::vector<uint64_t>& output);

    /**
     * @brief Inverse NTT (frequency domain -> time domain)
     *
     * @param input Input NTT coefficients [N]
     * @param output Output polynomial coefficients [N]
     */
    void Inverse(const std::vector<uint64_t>& input, std::vector<uint64_t>& output);

    /**
     * @brief Batch forward NTT (process multiple polynomials)
     *
     * @param inputs Vector of input polynomials, each [N]
     * @param outputs Vector of output NTT coefficients
     */
    void ForwardBatch(const std::vector<std::vector<uint64_t>>& inputs,
                      std::vector<std::vector<uint64_t>>& outputs);

    /**
     * @brief Batch inverse NTT
     */
    void InverseBatch(const std::vector<std::vector<uint64_t>>& inputs,
                      std::vector<std::vector<uint64_t>>& outputs);

    //--------------------------------------------------------------------------
    // In-Place Operations (GPU buffer interface)
    //--------------------------------------------------------------------------

    /**
     * @brief Forward NTT on GPU buffer (no CPU-GPU transfer)
     *
     * @param buffer Metal buffer containing polynomial data [batch_size * N]
     * @param batch_size Number of polynomials in buffer
     */
    void ForwardInPlace(id<MTLBuffer> buffer, uint32_t batch_size);

    /**
     * @brief Inverse NTT on GPU buffer
     */
    void InverseInPlace(id<MTLBuffer> buffer, uint32_t batch_size);

    //--------------------------------------------------------------------------
    // Accessors
    //--------------------------------------------------------------------------

    uint32_t GetN() const { return n_; }
    uint64_t GetQ() const { return q_; }
    uint32_t GetN1() const { return tile_config_.N1; }
    uint32_t GetN2() const { return tile_config_.N2; }
    TileConfig GetTileConfig() const { return tile_config_; }
    bool IsInitialized() const { return initialized_; }

private:
    //--------------------------------------------------------------------------
    // Initialization
    //--------------------------------------------------------------------------

    void Initialize();
    void PrecomputeTwiddles();
    void LoadShaders();
    void CreatePipelines();

    //--------------------------------------------------------------------------
    // Internal NTT Steps
    //--------------------------------------------------------------------------

    /**
     * @brief Step 1: Column NTTs (N2 parallel NTTs of size N1)
     */
    void ColumnNTTs(id<MTLBuffer> data, uint32_t batch_size, bool inverse);

    /**
     * @brief Step 2+3: Fused twiddle multiplication and transpose
     */
    void TwiddleAndTranspose(id<MTLBuffer> input, id<MTLBuffer> output,
                             uint32_t batch_size, bool inverse);

    /**
     * @brief Step 4: Row NTTs (N1 parallel NTTs of size N2)
     */
    void RowNTTs(id<MTLBuffer> data, uint32_t batch_size, bool inverse);

    /**
     * @brief Apply N^{-1} scaling for inverse NTT
     */
    void ScaleByNInverse(id<MTLBuffer> data, uint32_t batch_size);

    //--------------------------------------------------------------------------
    // Helper Methods
    //--------------------------------------------------------------------------

    uint64_t ModPow(uint64_t base, uint64_t exp, uint64_t mod);
    uint64_t ModInverse(uint64_t a, uint64_t mod);
    uint64_t ComputeBarrettConstant(uint64_t q);
    uint32_t Log2(uint32_t n);
    void WaitForCompletion();

    //--------------------------------------------------------------------------
    // Member Variables
    //--------------------------------------------------------------------------

    uint32_t n_;           // Ring dimension
    uint64_t q_;           // Prime modulus
    uint32_t log_n_;       // log2(n)
    TileConfig tile_config_;
    bool initialized_;

    // Metal resources
    id<MTLDevice> device_;
    id<MTLCommandQueue> command_queue_;
    id<MTLLibrary> library_;

    // Compute pipelines
    id<MTLComputePipelineState> column_ntt_pipeline_;
    id<MTLComputePipelineState> column_intt_pipeline_;
    id<MTLComputePipelineState> row_ntt_pipeline_;
    id<MTLComputePipelineState> row_intt_pipeline_;
    id<MTLComputePipelineState> twiddle_transpose_pipeline_;
    id<MTLComputePipelineState> inv_twiddle_transpose_pipeline_;
    id<MTLComputePipelineState> scale_pipeline_;

    // Precomputed twiddle factors (GPU buffers)
    id<MTLBuffer> column_twiddles_;        // Twiddles for N1-point NTT
    id<MTLBuffer> column_inv_twiddles_;    // Inverse twiddles
    id<MTLBuffer> row_twiddles_;           // Twiddles for N2-point NTT
    id<MTLBuffer> row_inv_twiddles_;       // Inverse twiddles
    id<MTLBuffer> transpose_twiddles_;     // omega^(i*j) for Step 2
    id<MTLBuffer> inv_transpose_twiddles_; // omega^(-i*j) for inverse

    // Precomputed Barrett constants
    id<MTLBuffer> column_twiddle_precon_;
    id<MTLBuffer> column_inv_twiddle_precon_;
    id<MTLBuffer> row_twiddle_precon_;
    id<MTLBuffer> row_inv_twiddle_precon_;

    // Scratch buffer for transpose intermediate
    id<MTLBuffer> scratch_buffer_;

    // Parameters buffer
    id<MTLBuffer> params_buffer_;
    FourStepNTTParams params_;
};

#else // !WITH_METAL

/**
 * @brief Stub class when Metal is not available
 */
class FourStepNTT {
public:
    FourStepNTT(uint32_t n, uint64_t q, void* device = nullptr) {
        (void)n; (void)q; (void)device;
        throw std::runtime_error("FourStepNTT requires Metal support. Compile with -DWITH_METAL=ON");
    }

    void Forward(const std::vector<uint64_t>& input, std::vector<uint64_t>& output) {
        (void)input; (void)output;
        throw std::runtime_error("Metal not available");
    }

    void Inverse(const std::vector<uint64_t>& input, std::vector<uint64_t>& output) {
        (void)input; (void)output;
        throw std::runtime_error("Metal not available");
    }

    uint32_t GetN() const { return 0; }
    uint64_t GetQ() const { return 0; }
    bool IsInitialized() const { return false; }
};

#endif // WITH_METAL

//==================================================================================
// Utility Functions
//==================================================================================

/**
 * @brief Check if Four-Step NTT is available (Metal support compiled in)
 */
inline bool IsFourStepNTTAvailable() {
#ifdef WITH_METAL
    return true;
#else
    return false;
#endif
}

/**
 * @brief Get recommended tile configuration for given N
 */
inline TileConfig GetRecommendedTileConfig(uint32_t N) {
    return TileConfig::ForN(N);
}

/**
 * @brief Validate N for Four-Step NTT
 *
 * Requirements:
 * - N must be power of 2
 * - N >= 1024 (otherwise standard NTT is faster)
 * - N <= 2^20 (practical limit for FHE)
 */
inline bool ValidateN(uint32_t N) {
    if (N < 1024) return false;
    if (N > (1u << 20)) return false;
    if ((N & (N - 1)) != 0) return false;  // Not power of 2
    return true;
}

/**
 * @brief Compute optimal batch size for given N and available GPU memory
 *
 * @param N Ring dimension
 * @param available_memory Available GPU memory in bytes
 * @return Optimal batch size
 */
inline uint32_t ComputeOptimalBatchSize(uint32_t N, size_t available_memory) {
    // Each polynomial needs N * 8 bytes (uint64_t)
    // Plus scratch space for transpose (~1.5x)
    size_t bytes_per_poly = N * sizeof(uint64_t) * 3;  // input + output + scratch

    uint32_t max_batch = static_cast<uint32_t>(available_memory / bytes_per_poly);

    // Clamp to reasonable values
    if (max_batch < 1) max_batch = 1;
    if (max_batch > 1024) max_batch = 1024;

    // Round down to power of 2 for better threadgroup scheduling
    uint32_t batch = 1;
    while (batch * 2 <= max_batch) batch *= 2;

    return batch;
}

} // namespace metal_backend
} // namespace lbcrypto

#endif // LBCRYPTO_MATH_HAL_MLX_FOUR_STEP_NTT_H
