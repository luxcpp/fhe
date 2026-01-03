//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2024, Lux Industries Inc
//
// All rights reserved.
//
// MLX GPU Backend for OpenFHE - Apple Silicon Acceleration
//
// This provides GPU-accelerated operations using Apple's MLX framework
// for significant performance improvements on Apple Silicon (M1/M2/M3/M4)
//==================================================================================

#ifndef LUX_FHE_MATH_HAL_MLX_BACKEND_H
#define LUX_FHE_MATH_HAL_MLX_BACKEND_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifdef WITH_MLX
#include <mlx/mlx.h>
#endif

namespace lux {
namespace mlx_backend {

/**
 * @brief Configuration for MLX GPU acceleration
 */
struct MLXConfig {
    std::string device_type = "gpu";     ///< "gpu" or "cpu"
    size_t batch_size = 64;              ///< Optimal batch size for NTT
    bool enable_caching = true;          ///< Cache twiddle factors on GPU
    size_t max_poly_degree = 32768;      ///< Maximum polynomial degree to support
    bool async_execution = true;         ///< Enable async GPU execution
};

/**
 * @brief Check if MLX backend is available
 * @return true if MLX is compiled in and GPU is available
 */
bool IsMLXAvailable();

/**
 * @brief Get MLX device name
 * @return Device name string (e.g., "Apple M3 Max GPU")
 */
std::string GetDeviceName();

/**
 * @brief Get current GPU memory usage
 * @return Memory usage in bytes
 */
size_t GetGPUMemoryUsage();

#ifdef WITH_MLX

namespace mx = mlx::core;

/**
 * @brief MLX-accelerated Number Theoretic Transform
 * 
 * Provides GPU-accelerated NTT/INTT operations for polynomial multiplication.
 * The NTT is the core operation in lattice cryptography and dominates runtime.
 */
class MLXNTT {
public:
    /**
     * @brief Initialize MLX NTT with parameters
     * @param n Ring dimension (power of 2)
     * @param q Modulus
     * @param config MLX configuration
     */
    explicit MLXNTT(uint64_t n, uint64_t q, const MLXConfig& config = MLXConfig{});
    ~MLXNTT();

    // Disable copy, allow move
    MLXNTT(const MLXNTT&) = delete;
    MLXNTT& operator=(const MLXNTT&) = delete;
    MLXNTT(MLXNTT&&) noexcept;
    MLXNTT& operator=(MLXNTT&&) noexcept;

    /**
     * @brief Forward NTT on GPU
     * @param input Input polynomial coefficients
     * @param output Output NTT coefficients (can be same as input for in-place)
     */
    void ForwardTransform(const std::vector<uint64_t>& input, std::vector<uint64_t>& output);

    /**
     * @brief Inverse NTT on GPU
     * @param input Input NTT coefficients
     * @param output Output polynomial coefficients
     */
    void InverseTransform(const std::vector<uint64_t>& input, std::vector<uint64_t>& output);

    /**
     * @brief Batch forward NTT on GPU (most efficient)
     * @param inputs Vector of input polynomials
     * @param outputs Vector of output polynomials
     */
    void ForwardTransformBatch(const std::vector<std::vector<uint64_t>>& inputs,
                                std::vector<std::vector<uint64_t>>& outputs);

    /**
     * @brief Batch inverse NTT on GPU
     */
    void InverseTransformBatch(const std::vector<std::vector<uint64_t>>& inputs,
                                std::vector<std::vector<uint64_t>>& outputs);

    /**
     * @brief Element-wise modular multiplication on GPU
     * @param a First polynomial in NTT domain
     * @param b Second polynomial in NTT domain
     * @param result Output polynomial in NTT domain
     */
    void ElementwiseMultMod(const std::vector<uint64_t>& a, 
                            const std::vector<uint64_t>& b,
                            std::vector<uint64_t>& result);

    /**
     * @brief Get ring dimension
     */
    uint64_t GetN() const { return n_; }

    /**
     * @brief Get modulus
     */
    uint64_t GetQ() const { return q_; }

    /**
     * @brief Check if using GPU
     */
    bool IsGPUEnabled() const { return gpu_enabled_; }

private:
    uint64_t n_;           ///< Ring dimension
    uint64_t q_;           ///< Modulus
    bool gpu_enabled_;     ///< Whether GPU is active
    MLXConfig config_;     ///< Configuration

    // Cached MLX arrays for twiddle factors (GPU path)
    mx::array twiddles_;         ///< Forward NTT twiddle factors
    mx::array inv_twiddles_;     ///< Inverse NTT twiddle factors
    mx::array n_inv_;            ///< 1/n mod q for inverse NTT
    
    // CPU-side twiddle factors for exact integer arithmetic
    std::vector<uint64_t> twiddle_factors_;      ///< Forward twiddles (w^0, w^1, ...)
    std::vector<uint64_t> inv_twiddle_factors_;  ///< Inverse twiddles (w^-0, w^-1, ...)
    uint64_t n_inv_val_;                         ///< n^(-1) mod q

    // Helper methods
    void PrecomputeTwiddles();
    mx::array ToMLXArray(const std::vector<uint64_t>& vec);
    void FromMLXArray(const mx::array& arr, std::vector<uint64_t>& vec);
    mx::array NTTCore(const mx::array& input, const mx::array& twiddles);
    mx::array INTTCore(const mx::array& input, const mx::array& inv_twiddles);
    
    // CPU NTT for exact modular arithmetic (GPU float32 loses precision)
    void NTTCpu(std::vector<uint64_t>& data, bool inverse);
};

/**
 * @brief MLX-accelerated polynomial operations for TFHE
 * 
 * Provides GPU-accelerated operations for blind rotation and bootstrapping.
 */
class MLXPolyOps {
public:
    /**
     * @brief Initialize with ring parameters
     * @param n Ring dimension
     * @param q Modulus
     * @param config MLX configuration
     */
    explicit MLXPolyOps(uint64_t n, uint64_t q, const MLXConfig& config = MLXConfig{});
    ~MLXPolyOps();

    /**
     * @brief Polynomial multiplication using NTT
     * @param a First polynomial
     * @param b Second polynomial
     * @param result Output polynomial (a * b mod X^n+1)
     */
    void PolyMult(const std::vector<uint64_t>& a,
                  const std::vector<uint64_t>& b,
                  std::vector<uint64_t>& result);

    /**
     * @brief Polynomial addition mod q
     */
    void PolyAdd(const std::vector<uint64_t>& a,
                 const std::vector<uint64_t>& b,
                 std::vector<uint64_t>& result);

    /**
     * @brief Polynomial subtraction mod q
     */
    void PolySub(const std::vector<uint64_t>& a,
                 const std::vector<uint64_t>& b,
                 std::vector<uint64_t>& result);

    /**
     * @brief Polynomial negation mod q
     */
    void PolyNeg(const std::vector<uint64_t>& a,
                 std::vector<uint64_t>& result);

    /**
     * @brief Automorphism X -> X^k (Galois automorphism)
     * @param a Input polynomial
     * @param k Automorphism index (odd)
     * @param result Output polynomial
     */
    void Automorphism(const std::vector<uint64_t>& a,
                      uint64_t k,
                      std::vector<uint64_t>& result);

    /**
     * @brief RGSW external product (key operation for blind rotation)
     * @param ct RLWE ciphertext [a, b]
     * @param rgsw RGSW ciphertext (encryption of m)
     * @param result Output RLWE ciphertext (ct * m)
     */
    void RGSWExternalProduct(const std::vector<std::vector<uint64_t>>& ct,
                             const std::vector<std::vector<uint64_t>>& rgsw,
                             std::vector<std::vector<uint64_t>>& result);

private:
    std::unique_ptr<MLXNTT> ntt_;
    uint64_t n_;
    uint64_t q_;
    bool gpu_enabled_;
};

/**
 * @brief MLX-accelerated blind rotation for TFHE bootstrapping
 */
class MLXBlindRotation {
public:
    /**
     * @brief Initialize with TFHE parameters
     * @param n LWE dimension
     * @param N Ring dimension
     * @param q LWE modulus
     * @param Q Ring modulus
     */
    MLXBlindRotation(uint64_t n, uint64_t N, uint64_t q, uint64_t Q,
                     const MLXConfig& config = MLXConfig{});
    ~MLXBlindRotation();

    /**
     * @brief Perform blind rotation
     * @param acc Initial accumulator (RLWE encryption of test polynomial)
     * @param lwe_ct LWE ciphertext [a, b]
     * @param bsk Bootstrap key (RGSW encryptions of s_i)
     * @param result Output accumulator
     */
    void Evaluate(const std::vector<std::vector<uint64_t>>& acc,
                  const std::vector<uint64_t>& lwe_ct,
                  const std::vector<std::vector<std::vector<uint64_t>>>& bsk,
                  std::vector<std::vector<uint64_t>>& result);

    /**
     * @brief Batch blind rotation (for parallel bootstraps)
     */
    void EvaluateBatch(const std::vector<std::vector<std::vector<uint64_t>>>& accs,
                       const std::vector<std::vector<uint64_t>>& lwe_cts,
                       const std::vector<std::vector<std::vector<uint64_t>>>& bsk,
                       std::vector<std::vector<std::vector<uint64_t>>>& results);

private:
    std::unique_ptr<MLXPolyOps> poly_ops_;
    uint64_t n_;   // LWE dimension
    uint64_t N_;   // Ring dimension
    [[maybe_unused]] uint64_t q_;   // LWE modulus (for future key switching)
    [[maybe_unused]] uint64_t Q_;   // Ring modulus (for future modulus switching)
};

#endif // WITH_MLX

// =============================================================================
// C-style API for Go CGO (all platforms)
// =============================================================================

/**
 * @brief Forward NTT on flat data array
 * @param data Flat array [batch * N] (modified in-place)
 * @param N Ring dimension
 * @param Q Modulus
 * @param batch Number of polynomials
 * @return 0 on success, -1 on error
 */
int ntt_forward(uint64_t* data, uint32_t N, uint64_t Q, uint32_t batch);

/**
 * @brief Inverse NTT on flat data array
 */
int ntt_inverse(uint64_t* data, uint32_t N, uint64_t Q, uint32_t batch);

/**
 * @brief Pointwise modular multiplication
 */
int pointwise_mul(uint64_t* result, const uint64_t* a, const uint64_t* b,
                  uint32_t N, uint64_t Q, uint32_t batch);

/**
 * @brief Clear internal caches
 */
void clear_cache();

} // namespace mlx_backend
} // namespace lux

#endif // LUX_FHE_MATH_HAL_MLX_BACKEND_H
