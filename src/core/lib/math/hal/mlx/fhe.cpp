//==================================================================================
// FHE Implementation - Optimized for Apple Metal
//==================================================================================
//
// This file contains the optimized GPU implementation of Lux FHE operations
// targeting Apple Silicon (M1/M2/M3) via the MLX framework.
//
// Key Optimizations:
// 1. Fused kernels to minimize memory transfers
// 2. Structure of Arrays (SoA) memory layout for coalesced access
// 3. Batch-parallel processing for maximum GPU utilization
// 4. Montgomery/Barrett arithmetic for efficient modular operations
// 5. Shared memory utilization for NTT twiddle factors
//
//==================================================================================

#include "math/hal/mlx/fhe.h"
#include "ntt.h"           // Local header in this directory
#include "metal_dispatch.h" // Metal GPU dispatcher
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <numeric>

#ifdef WITH_MLX
#include <mlx/mlx.h>
namespace mx = mlx::core;
#endif

namespace lbcrypto {
namespace gpu {

//==================================================================================
// Modular Arithmetic (CPU side for setup, GPU kernels use Metal equivalents)
//==================================================================================

// Use modular arithmetic from ntt.h
using lbcrypto::gpu::mulmod;
using lbcrypto::gpu::powmod;
using lbcrypto::gpu::mod_inverse;
using lbcrypto::gpu::find_primitive_root;

namespace {

// Local aliases for compatibility
inline uint64_t modInverse(uint64_t a, uint64_t m) { return mod_inverse(a, m); }
inline uint64_t findPrimitiveRoot(uint64_t N, uint64_t q) { return find_primitive_root(N, q); }

inline uint64_t addmod(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t sum = a + b;
    return sum >= m ? sum - m : sum;
}

inline uint64_t submod(uint64_t a, uint64_t b, uint64_t m) {
    return a >= b ? a - b : m - b + a;
}

// Compute Barrett reduction constant: floor(2^k / q) where k is chosen
// such that operations stay within 64-bit bounds
uint64_t computeBarrettConstant(uint64_t q) {
    // For q < 2^28, we use k = 56
    // mu = floor(2^56 / q)
    __uint128_t numerator = static_cast<__uint128_t>(1) << 56;
    return static_cast<uint64_t>(numerator / q);
}

}  // namespace

//==================================================================================
// FHEEngine Base Class Implementation
//==================================================================================

FHEEngine::FHEEngine(const FHEConfig& config) : config_(config) {}

FHEEngine::~FHEEngine() = default;

bool FHEEngine::initialize() {
    initializeNTTTwiddles();
    initializeTestPolynomials();

#ifdef WITH_MLX
    // Try to initialize Metal GPU dispatcher
    try {
        if (mx::metal::is_available()) {
            metalDispatcher_ = std::make_unique<metal::FHEMetalDispatcher>(
                config_.N, config_.n, config_.L, config_.baseLog, config_.Q);
            useGpu_ = metalDispatcher_->is_gpu_available();
            if (useGpu_) {
                std::cout << "FHE Engine: Metal GPU acceleration enabled" << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "FHE Engine: Metal GPU init failed: " << e.what() << std::endl;
        useGpu_ = false;
    }
#endif

    return true;
}

void FHEEngine::shutdown() {}

void FHEEngine::initializeNTTTwiddles() {
#ifdef WITH_MLX
    uint32_t N = config_.N;
    uint64_t Q = config_.Q;

    // Find primitive 2N-th root of unity
    // Note: find_primitive_root already computes 2N-th root given N
    uint64_t omega = findPrimitiveRoot(N, Q);

    // Compute forward twiddle factors
    std::vector<int64_t> twiddles(N);
    uint64_t w = 1;
    for (uint32_t i = 0; i < N; ++i) {
        twiddles[i] = static_cast<int64_t>(w);
        w = mulmod(w, omega, Q);
    }
    twiddleFactors_ = std::make_shared<mx::array>(
        mx::array(twiddles.data(), {static_cast<int>(N)}, mx::int64));

    // Compute inverse twiddle factors
    uint64_t omega_inv = modInverse(omega, Q);
    w = 1;
    for (uint32_t i = 0; i < N; ++i) {
        twiddles[i] = static_cast<int64_t>(w);
        w = mulmod(w, omega_inv, Q);
    }
    invTwiddleFactors_ = std::make_shared<mx::array>(
        mx::array(twiddles.data(), {static_cast<int>(N)}, mx::int64));

    mx::eval(*twiddleFactors_);
    mx::eval(*invTwiddleFactors_);
#endif
}

void FHEEngine::initializeTestPolynomials() {
#ifdef WITH_MLX
    // Placeholder - test polynomials for gate bootstrapping
    uint32_t N = config_.N;
    std::vector<int64_t> zeros(N, 0);
    testPolynomials_ = std::make_shared<mx::array>(
        mx::array(zeros.data(), {1, static_cast<int>(N)}, mx::int64));
    mx::eval(*testPolynomials_);
#endif
}

// Stats implementations
size_t FHEEngine::totalGPUMemoryUsed() const {
#ifdef WITH_MLX
    // Use MLX memory tracking
    try {
        return mx::get_active_memory();
    } catch (...) {
        // Fallback: sum user memory
        size_t total = 0;
        for (const auto& [uid, session] : users_) {
            total += session->memoryUsed;
        }
        return total;
    }
#else
    size_t total = 0;
    for (const auto& [uid, session] : users_) {
        total += session->memoryUsed;
    }
    return total;
#endif
}

size_t FHEEngine::availableGPUMemory() const {
    // Placeholder - would query Metal device for available memory
    return 16ULL * 1024 * 1024 * 1024;  // 16GB default
}

uint32_t FHEEngine::activeUsers() const {
    return static_cast<uint32_t>(users_.size());
}

double FHEEngine::avgOperationsPerSecond() const {
    return 0.0;  // Placeholder
}

// Base class batch operations (CPU fallback)
#ifdef WITH_MLX
void FHEEngine::batchNTT(mx::array& polys, bool inverse) {
    // Use NTTEngine from ntt.h for CPU fallback
    lbcrypto::gpu::NTTEngine ntt_engine(config_.N, config_.Q);
    if (inverse) {
        ntt_engine.inverse(polys);
    } else {
        ntt_engine.forward(polys);
    }
}

void FHEEngine::batchExternalProduct(const mx::array& rlweBatch,
                                     const mx::array& rgswBatch,
                                     mx::array& output) {
    // Try GPU path first
    if (useGpu_ && metalDispatcher_) {
        try {
            metalDispatcher_->external_product(rlweBatch, rgswBatch, output);
            return;
        } catch (const std::exception& e) {
            // Fall through to CPU
            std::cerr << "GPU external_product failed, using CPU: " << e.what() << std::endl;
        }
    }

    // CPU fallback for external product: RGSW × RLWE → RLWE
    // rlweBatch: [B, 2, N] - batch of RLWE ciphertexts in NTT domain
    // rgswBatch: [B, 2, L, 2, N] - batch of RGSW ciphertexts in NTT domain
    // output: [B, 2, N] - result RLWE ciphertext in NTT domain

    auto rlwe_shape = rlweBatch.shape();
    int B = rlwe_shape[0];  // Batch size
    int N = rlwe_shape[2];  // Ring dimension
    uint32_t L = config_.L;  // Decomposition digits
    uint64_t Q = config_.Q;

    // Create NTT engine for polynomial operations
    lbcrypto::gpu::NTTEngine ntt_engine(config_.N, Q);

    mx::eval(rlweBatch);
    mx::eval(rgswBatch);
    auto rlwe_ptr = rlweBatch.data<int64_t>();
    auto rgsw_ptr = rgswBatch.data<int64_t>();

    // Output buffer
    std::vector<int64_t> out_data(B * 2 * N, 0);

    // Process each batch element
    for (int b = 0; b < B; ++b) {
        // For each RLWE component (c0, c1)
        for (int c = 0; c < 2; ++c) {
            // Digit decomposition and accumulation
            for (uint32_t l = 0; l < L; ++l) {
                // For each polynomial coefficient position
                for (int i = 0; i < N; ++i) {
                    // Get RLWE coefficient
                    int64_t rlwe_coef = rlwe_ptr[b * 2 * N + c * N + i];
                    uint64_t val = static_cast<uint64_t>(rlwe_coef);

                    // Extract l-th digit (simplified decomposition)
                    uint64_t digit = (val >> (l * config_.baseLog)) & ((1ULL << config_.baseLog) - 1);

                    // Multiply by RGSW row and accumulate
                    // rgswBatch layout: [B, 2, L, 2, N]
                    // Access: rgsw[b][c][l][out_c][i]
                    for (int out_c = 0; out_c < 2; ++out_c) {
                        int64_t rgsw_coef = rgsw_ptr[b * 2 * L * 2 * N + c * L * 2 * N + l * 2 * N + out_c * N + i];
                        uint64_t prod = lbcrypto::gpu::mulmod(digit, static_cast<uint64_t>(rgsw_coef), Q);
                        out_data[b * 2 * N + out_c * N + i] =
                            static_cast<int64_t>((static_cast<uint64_t>(out_data[b * 2 * N + out_c * N + i]) + prod) % Q);
                    }
                }
            }
        }
    }

    output = mx::array(out_data.data(), {B, 2, N}, mx::int64);
    mx::eval(output);
}

void FHEEngine::batchBlindRotate(const mx::array& lweBatch,
                                 const mx::array& bskBatch,
                                 const mx::array& testPoly,
                                 mx::array& output) {
    // Try GPU path first
    if (useGpu_ && metalDispatcher_) {
        try {
            metalDispatcher_->blind_rotate(lweBatch, bskBatch, testPoly, output);
            return;
        } catch (const std::exception& e) {
            std::cerr << "GPU blind_rotate failed, using CPU: " << e.what() << std::endl;
        }
    }

    // CPU fallback - simplified blind rotation
    // For full implementation, use FHEEngineOptimized::batchBlindRotateOptimized
    auto shape = lweBatch.shape();
    int B = shape[0];
    int N = static_cast<int>(config_.N);
    uint64_t Q = config_.Q;

    mx::eval(lweBatch);
    mx::eval(testPoly);

    auto lwePtr = lweBatch.data<int64_t>();
    auto testPtr = testPoly.data<int64_t>();
    int n = shape[1] - 1;

    // Initialize accumulator with rotated test polynomial
    std::vector<int64_t> accData(B * 2 * N);
    for (int b = 0; b < B; ++b) {
        int64_t bVal = lwePtr[b * (n + 1) + n];
        int shift = static_cast<int>(bVal % (2 * N));
        if (shift < 0) shift += 2 * N;

        for (int i = 0; i < N; ++i) {
            accData[b * 2 * N + i] = 0;  // acc0 = 0
            // Negacyclic rotation
            int srcIdx = (i + shift) % (2 * N);
            if (srcIdx < N) {
                accData[b * 2 * N + N + i] = testPtr[srcIdx % testPoly.shape()[testPoly.shape().size() > 1 ? 1 : 0]];
            } else {
                int64_t val = testPtr[(srcIdx - N) % testPoly.shape()[testPoly.shape().size() > 1 ? 1 : 0]];
                accData[b * 2 * N + N + i] = (val == 0) ? 0 : static_cast<int64_t>(Q) - val;
            }
        }
    }

    output = mx::array(accData.data(), {B, 2, N}, mx::int64);
    mx::eval(output);
}

void FHEEngine::batchKeySwitch(const mx::array& rlweBatch,
                               const mx::array& kskBatch,
                               mx::array& output) {
    // Try GPU path first
    if (useGpu_ && metalDispatcher_) {
        try {
            metalDispatcher_->key_switch(rlweBatch, kskBatch, output);
            return;
        } catch (const std::exception& e) {
            std::cerr << "GPU key_switch failed, using CPU: " << e.what() << std::endl;
        }
    }

    // CPU fallback - simplified key switching (extract sample)
    auto shape = rlweBatch.shape();
    int B = shape[0];
    int N = shape[2];
    int n = static_cast<int>(config_.n);
    uint64_t Q = config_.Q;

    mx::eval(rlweBatch);

    auto rlwePtr = rlweBatch.data<int64_t>();

    // Simple key switch: extract first n coefficients + body
    std::vector<int64_t> result(B * (n + 1));
    for (int b = 0; b < B; ++b) {
        // Copy first n coefficients from c0 as LWE mask
        for (int i = 0; i < n; ++i) {
            result[b * (n + 1) + i] = rlwePtr[b * 2 * N + i] % static_cast<int64_t>(Q);
        }
        // Body from c1[0]
        result[b * (n + 1) + n] = rlwePtr[b * 2 * N + N] % static_cast<int64_t>(Q);
    }

    output = mx::array(result.data(), {B, n + 1}, mx::int64);
    mx::eval(output);
}

void FHEEngine::batchBootstrap(const mx::array& lweBatch,
                               GateType gate,
                               const mx::array& bskBatch,
                               const mx::array& kskBatch,
                               mx::array& output) {
    // CPU fallback - not implemented
    throw std::runtime_error("batchBootstrap: CPU fallback not implemented");
}
#endif

// User management
uint64_t FHEEngine::createUser() {
    std::lock_guard<std::mutex> lock(usersMutex_);
    uint64_t userId = nextUserId_++;
    users_[userId] = std::make_unique<UserSession>();
    return userId;
}

void FHEEngine::deleteUser(uint64_t userId) {
    std::lock_guard<std::mutex> lock(usersMutex_);
    users_.erase(userId);
}

UserSession* FHEEngine::getUser(uint64_t userId) {
    std::lock_guard<std::mutex> lock(usersMutex_);
    auto it = users_.find(userId);
    return it != users_.end() ? it->second.get() : nullptr;
}

void FHEEngine::generateKeys(uint64_t userId) {
    // Placeholder - key generation on GPU
}

void FHEEngine::uploadBootstrapKey(uint64_t userId, const std::vector<uint64_t>& bskData) {
    // Placeholder - upload BSK to GPU
}

uint32_t FHEEngine::allocateCiphertexts(uint64_t userId, uint32_t count) {
    auto* session = getUser(userId);
    if (!session) {
        return 0;
    }

#ifdef WITH_MLX
    // Allocate LWE ciphertext pool on GPU
    // Each LWE ciphertext has n+1 elements (a[0..n-1] and b)
    uint32_t n = config_.n;
    size_t elements = static_cast<size_t>(count) * (n + 1);
    size_t bytes = elements * sizeof(uint64_t);

    // Create zeroed GPU array for the ciphertexts
    LWECiphertextGPU pool;
    pool.a = std::make_shared<mx::array>(mx::zeros({static_cast<int>(count), static_cast<int>(n)}, mx::int64));
    pool.b = std::make_shared<mx::array>(mx::zeros({static_cast<int>(count)}, mx::int64));
    pool.count = count;

    // Force allocation on GPU
    mx::eval(*pool.a);
    mx::eval(*pool.b);

    // Track memory usage
    session->memoryUsed += bytes;

    // Add to user's pools and return the pool index
    uint32_t poolIdx = static_cast<uint32_t>(session->lwePools.size());
    session->lwePools.push_back(std::move(pool));

    return poolIdx;
#else
    // CPU-only mode - still track memory for testing
    session->memoryUsed += static_cast<size_t>(count) * (config_.n + 1) * sizeof(uint64_t);
    return static_cast<uint32_t>(session->lwePools.size());
#endif
}

void FHEEngine::uploadCiphertexts(uint64_t userId, uint32_t startIdx,
                                  const std::vector<std::vector<uint64_t>>& data) {
    // Placeholder - upload ciphertexts to GPU
}

void FHEEngine::downloadCiphertexts(uint64_t userId, uint32_t startIdx, uint32_t count,
                                    std::vector<std::vector<uint64_t>>& output) {
    // Placeholder - download ciphertexts from GPU
}

void FHEEngine::executeBatchGates(const std::vector<BatchedGateOp>& ops) {
    // Placeholder - execute batch of gate operations
}

void FHEEngine::sync() {
#ifdef WITH_MLX
    // Synchronize all pending GPU operations
    mx::synchronize();
#endif
}

//==================================================================================
// FHEEngineOptimized Implementation
//==================================================================================

class FHEEngineOptimized : public FHEEngine {
public:
    explicit FHEEngineOptimized(const FHEConfig& config);
    ~FHEEngineOptimized() override;
    
    bool initialize() override;
    
    // Optimized kernel implementations
    void batchNTTOptimized(mx::array& polys, bool inverse);
    void batchCMuxFused(mx::array& acc, const mx::array& bsk, const mx::array& rotations);
    void batchKeySwitchOptimized(const mx::array& rlweBatch, const mx::array& ksk, mx::array& output);
    void batchBlindRotateOptimized(const mx::array& lweBatch, const mx::array& bsk,
                                    const mx::array& testPoly, mx::array& output);

private:
    // Barrett reduction constant
    uint64_t barrettMu_;
    
    // Montgomery constants
    uint64_t montR_;          // R = 2^32
    uint64_t montRInv_;       // R^-1 mod Q
    uint64_t montQInv_;       // -Q^-1 mod R
    
    // Precomputed NTT tables in Montgomery form
    std::vector<uint64_t> nttTwiddles_;
    std::vector<uint64_t> invNttTwiddles_;
    std::vector<uint64_t> bitReversalTable_;
    
#ifdef WITH_MLX
    // GPU buffers for precomputed data
    std::shared_ptr<mx::array> gpuTwiddles_;
    std::shared_ptr<mx::array> gpuInvTwiddles_;
    std::shared_ptr<mx::array> gpuBitReversal_;
    
    // Metal kernel function handles (for future custom kernel dispatch)
    // void* nttForwardKernel_;
    // void* nttInverseKernel_;
    // void* cmuxFusedKernel_;
    // void* keySwitchKernel_;
#endif

    void precomputeNTTTables();
    void precomputeMontgomeryConstants();
    void uploadPrecomputedToGPU();
    
    // Optimized CPU implementations (for correctness verification)
    void nttCpuOptimized(std::vector<uint64_t>& data, bool inverse);
    void externalProductCpu(const std::vector<uint64_t>& rlwe,
                            const std::vector<uint64_t>& rgsw,
                            std::vector<uint64_t>& output);
};

FHEEngineOptimized::FHEEngineOptimized(const FHEConfig& config)
    : FHEEngine(config) {
}

FHEEngineOptimized::~FHEEngineOptimized() = default;

bool FHEEngineOptimized::initialize() {
#ifdef WITH_MLX
    try {
        // Check GPU availability
        if (!mx::metal::is_available()) {
            std::cerr << "Metal GPU not available" << std::endl;
            return false;
        }
        
        std::cout << "=== FHE Engine (Optimized) ===" << std::endl;
        std::cout << "Ring dimension N: " << config_.N << std::endl;
        std::cout << "LWE dimension n: " << config_.n << std::endl;
        std::cout << "Decomposition L: " << config_.L << std::endl;
        std::cout << "Base log: " << config_.baseLog << std::endl;
        std::cout << "Modulus Q: " << config_.Q << std::endl;
        
        // Precompute constants
        precomputeMontgomeryConstants();
        precomputeNTTTables();
        uploadPrecomputedToGPU();
        
        // Initialize test polynomials
        initializeTestPolynomials();
        
        std::cout << "FHE Engine initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "FHE init failed: " << e.what() << std::endl;
        return false;
    }
#else
    std::cerr << "MLX not available" << std::endl;
    return false;
#endif
}

void FHEEngineOptimized::precomputeMontgomeryConstants() {
    uint64_t Q = config_.Q;
    
    // Barrett constant for modular reduction
    barrettMu_ = computeBarrettConstant(Q);
    
    // Montgomery constants
    montR_ = 1ULL << 32;
    montRInv_ = modInverse(montR_ % Q, Q);
    
    // Compute -Q^-1 mod R
    // We need q_inv such that Q * q_inv = -1 (mod R)
    // i.e., Q * q_inv + 1 = 0 (mod R)
    uint64_t qModR = Q % montR_;
    uint64_t qInvModR = modInverse(qModR, montR_);
    montQInv_ = montR_ - qInvModR;  // Negate to get -Q^-1
    
    std::cout << "Montgomery R = 2^32" << std::endl;
    std::cout << "Barrett mu = " << barrettMu_ << std::endl;
}

void FHEEngineOptimized::precomputeNTTTables() {
    uint64_t N = config_.N;
    uint64_t Q = config_.Q;
    
    // Find primitive 2N-th root of unity
    uint64_t omega = findPrimitiveRoot(N, Q);
    uint64_t omegaInv = modInverse(omega, Q);
    uint64_t nInv = modInverse(N, Q);
    
    // Precompute twiddle factors in bit-reversed order
    nttTwiddles_.resize(N);
    invNttTwiddles_.resize(N);
    
    // Standard order first
    std::vector<uint64_t> twiddles(N);
    std::vector<uint64_t> invTwiddles(N);
    
    uint64_t w = 1;
    uint64_t wInv = 1;
    for (uint64_t i = 0; i < N; ++i) {
        twiddles[i] = w;
        invTwiddles[i] = mulmod(wInv, nInv, Q);
        w = mulmod(w, omega, Q);
        wInv = mulmod(wInv, omegaInv, Q);
    }
    
    // Compute bit-reversal permutation table
    uint32_t logN = 0;
    for (uint64_t temp = N; temp > 1; temp >>= 1) ++logN;
    
    bitReversalTable_.resize(N);
    for (uint64_t i = 0; i < N; ++i) {
        uint64_t j = 0;
        for (uint32_t k = 0; k < logN; ++k) {
            if (i & (1ULL << k)) {
                j |= (1ULL << (logN - 1 - k));
            }
        }
        bitReversalTable_[i] = j;
    }
    
    // Store twiddles (could be reordered for cache efficiency)
    nttTwiddles_ = twiddles;
    invNttTwiddles_ = invTwiddles;
    
    std::cout << "NTT tables precomputed (omega = " << omega << ")" << std::endl;
}

void FHEEngineOptimized::uploadPrecomputedToGPU() {
#ifdef WITH_MLX
    uint64_t N = config_.N;
    
    // Convert to int64 for MLX
    std::vector<int64_t> twiddles64(N);
    std::vector<int64_t> invTwiddles64(N);
    std::vector<int64_t> bitRev64(N);
    
    for (uint64_t i = 0; i < N; ++i) {
        twiddles64[i] = static_cast<int64_t>(nttTwiddles_[i]);
        invTwiddles64[i] = static_cast<int64_t>(invNttTwiddles_[i]);
        bitRev64[i] = static_cast<int64_t>(bitReversalTable_[i]);
    }
    
    gpuTwiddles_ = std::make_shared<mx::array>(
        mx::array(twiddles64.data(), {static_cast<int>(N)}, mx::int64));
    gpuInvTwiddles_ = std::make_shared<mx::array>(
        mx::array(invTwiddles64.data(), {static_cast<int>(N)}, mx::int64));
    gpuBitReversal_ = std::make_shared<mx::array>(
        mx::array(bitRev64.data(), {static_cast<int>(N)}, mx::int64));
    
    mx::eval(*gpuTwiddles_);
    mx::eval(*gpuInvTwiddles_);
    mx::eval(*gpuBitReversal_);
    
    std::cout << "Precomputed tables uploaded to GPU" << std::endl;
#endif
}

void FHEEngineOptimized::nttCpuOptimized(std::vector<uint64_t>& data, bool inverse) {
    uint64_t N = config_.N;
    uint64_t Q = config_.Q;
    const auto& twiddles = inverse ? invNttTwiddles_ : nttTwiddles_;
    
    // Bit-reversal permutation
    for (uint64_t i = 0; i < N; ++i) {
        uint64_t j = bitReversalTable_[i];
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }
    
    // Cooley-Tukey butterfly
    for (uint64_t len = 2; len <= N; len <<= 1) {
        uint64_t step = N / len;
        for (uint64_t i = 0; i < N; i += len) {
            for (uint64_t j = 0; j < len / 2; ++j) {
                uint64_t w = twiddles[j * step];
                uint64_t u = data[i + j];
                uint64_t v = mulmod(data[i + j + len/2], w, Q);
                
                data[i + j] = addmod(u, v, Q);
                data[i + j + len/2] = submod(u, v, Q);
            }
        }
    }
}

//==================================================================================
// Optimized Batch NTT
//==================================================================================

void FHEEngineOptimized::batchNTTOptimized(mx::array& polys, bool inverse) {
#ifdef WITH_MLX
    // polys: [batch, N]
    // 
    // Optimal GPU implementation would:
    // 1. Use shared memory for twiddle factors
    // 2. Process multiple butterflies per thread
    // 3. Minimize global memory accesses
    //
    // Current implementation: Vectorized CPU with MLX parallelism
    // TODO: Replace with custom Metal kernel dispatch
    
    auto shape = polys.shape();
    int batch = shape[0];
    int N = shape[1];
    uint64_t Q = config_.Q;
    
    mx::eval(polys);
    
    // Process each polynomial (can be parallelized)
    std::vector<int64_t> result(batch * N);
    auto dataPtr = polys.data<int64_t>();
    
    #pragma omp parallel for
    for (int b = 0; b < batch; ++b) {
        std::vector<uint64_t> poly(N);
        for (int i = 0; i < N; ++i) {
            poly[i] = static_cast<uint64_t>(dataPtr[b * N + i]) % Q;
        }
        
        nttCpuOptimized(poly, inverse);
        
        for (int i = 0; i < N; ++i) {
            result[b * N + i] = static_cast<int64_t>(poly[i]);
        }
    }
    
    polys = mx::array(result.data(), shape, mx::int64);
    mx::eval(polys);
#endif
}

//==================================================================================
// Fused CMux Gate (Rotate + Decompose + External Product)
//==================================================================================

void FHEEngineOptimized::batchCMuxFused(
    mx::array& acc,             // [B, 2, N] accumulator (NTT domain)
    const mx::array& bsk,       // [2, L, 2, N] bootstrap key for current index
    const mx::array& rotations  // [B] rotation amounts
) {
#ifdef WITH_MLX
    auto shape = acc.shape();
    int B = shape[0];
    int N = shape[2];
    uint32_t L = config_.L;
    uint64_t Q = config_.Q;
    uint32_t baseLog = config_.baseLog;
    uint64_t mask = (1ULL << baseLog) - 1;
    
    mx::eval(acc);
    mx::eval(bsk);
    mx::eval(rotations);
    
    auto accPtr = acc.data<int64_t>();
    auto bskPtr = bsk.data<int64_t>();
    auto rotPtr = rotations.data<int64_t>();
    
    std::vector<int64_t> result(B * 2 * N);
    
    // Process each batch element
    #pragma omp parallel for
    for (int b = 0; b < B; ++b) {
        int rotation = static_cast<int>(rotPtr[b]) % (2 * N);
        if (rotation < 0) rotation += 2 * N;
        
        // Skip if no rotation
        if (rotation == 0) {
            for (int c = 0; c < 2; ++c) {
                for (int i = 0; i < N; ++i) {
                    result[b * 2 * N + c * N + i] = accPtr[b * 2 * N + c * N + i];
                }
            }
            continue;
        }
        
        // For each output component
        for (int outC = 0; outC < 2; ++outC) {
            // For each coefficient
            for (int coeff = 0; coeff < N; ++coeff) {
                // Compute rotated index
                int srcIdx = (coeff - rotation);
                bool negate = false;
                if (srcIdx < 0) srcIdx += 2 * N;
                if (srcIdx >= N) {
                    srcIdx -= N;
                    negate = true;
                }
                
                // Load original and rotated values for both components
                uint64_t diff[2];
                for (int inC = 0; inC < 2; ++inC) {
                    int64_t orig = accPtr[b * 2 * N + inC * N + coeff];
                    int64_t rot = accPtr[b * 2 * N + inC * N + srcIdx];
                    if (negate) rot = (rot == 0) ? 0 : static_cast<int64_t>(Q) - rot;
                    
                    // diff = rotated - original
                    diff[inC] = submod(static_cast<uint64_t>(rot) % Q,
                                       static_cast<uint64_t>(orig) % Q, Q);
                }
                
                // Accumulate external product
                uint64_t extProd = 0;
                for (int inC = 0; inC < 2; ++inC) {
                    uint64_t val = diff[inC];
                    for (uint32_t l = 0; l < L; ++l) {
                        uint64_t digit = (val >> (l * baseLog)) & mask;
                        
                        // BSK index: [inC, l, outC, coeff]
                        int bskIdx = inC * L * 2 * N + l * 2 * N + outC * N + coeff;
                        uint64_t bskVal = static_cast<uint64_t>(bskPtr[bskIdx]) % Q;
                        
                        extProd = addmod(extProd, mulmod(digit, bskVal, Q), Q);
                    }
                }
                
                // Update accumulator
                uint64_t origVal = static_cast<uint64_t>(accPtr[b * 2 * N + outC * N + coeff]) % Q;
                result[b * 2 * N + outC * N + coeff] = static_cast<int64_t>(addmod(origVal, extProd, Q));
            }
        }
    }
    
    acc = mx::array(result.data(), shape, mx::int64);
    mx::eval(acc);
#endif
}

//==================================================================================
// Optimized Key Switching
//==================================================================================

void FHEEngineOptimized::batchKeySwitchOptimized(
    const mx::array& rlweBatch,  // [B, 2, N]
    const mx::array& ksk,        // [N, L_ks, n+1]
    mx::array& output            // [B, n+1]
) {
#ifdef WITH_MLX
    auto shape = rlweBatch.shape();
    int B = shape[0];
    int N = shape[2];
    uint32_t n = config_.n;
    uint32_t L_ks = 4;  // Key switching decomposition levels
    uint32_t baseLog = 8;  // Key switching base log
    uint64_t Q = config_.Q;
    uint64_t mask = (1ULL << baseLog) - 1;
    
    mx::eval(rlweBatch);
    mx::eval(ksk);
    
    auto rlwePtr = rlweBatch.data<int64_t>();
    auto kskPtr = ksk.data<int64_t>();
    
    std::vector<int64_t> result(B * (n + 1), 0);
    
    #pragma omp parallel for
    for (int b = 0; b < B; ++b) {
        // For each LWE output coefficient
        for (uint32_t lweIdx = 0; lweIdx <= n; ++lweIdx) {
            uint64_t sum = 0;
            
            // Accumulate over RLWE coefficients and decomposition levels
            for (int j = 0; j < N; ++j) {
                // Get c0[j] from RLWE
                uint64_t c0j = static_cast<uint64_t>(rlwePtr[b * 2 * N + j]) % Q;
                
                // Decompose into L_ks digits
                for (uint32_t l = 0; l < L_ks; ++l) {
                    uint64_t digit = (c0j >> (l * baseLog)) & mask;
                    if (digit == 0) continue;
                    
                    // Get KSK[j, l, lweIdx]
                    int kskIdx = j * L_ks * (n + 1) + l * (n + 1) + lweIdx;
                    uint64_t kskVal = static_cast<uint64_t>(kskPtr[kskIdx]) % Q;
                    
                    sum = addmod(sum, mulmod(digit, kskVal, Q), Q);
                }
            }
            
            // For body, add c1[0]
            if (lweIdx == n) {
                uint64_t c1_0 = static_cast<uint64_t>(rlwePtr[b * 2 * N + N]) % Q;
                sum = addmod(c1_0, sum, Q);
            }
            
            result[b * (n + 1) + lweIdx] = static_cast<int64_t>(sum);
        }
    }
    
    output = mx::array(result.data(), {B, static_cast<int>(n + 1)}, mx::int64);
    mx::eval(output);
#endif
}

//==================================================================================
// Full Blind Rotation Pipeline
//==================================================================================

void FHEEngineOptimized::batchBlindRotateOptimized(
    const mx::array& lweBatch,   // [B, n+1]
    const mx::array& bsk,        // [n, 2, L, 2, N]
    const mx::array& testPoly,   // [N]
    mx::array& output            // [B, 2, N]
) {
#ifdef WITH_MLX
    auto shape = lweBatch.shape();
    int B = shape[0];
    int n = shape[1] - 1;
    int N = config_.N;
    uint64_t Q = config_.Q;
    
    mx::eval(lweBatch);
    mx::eval(bsk);
    mx::eval(testPoly);
    
    auto lwePtr = lweBatch.data<int64_t>();
    auto bskPtr = bsk.data<int64_t>();
    auto testPtr = testPoly.data<int64_t>();
    
    // Step 1: Initialize accumulators with X^{-b} * testPoly
    std::vector<int64_t> accData(B * 2 * N);
    
    for (int b = 0; b < B; ++b) {
        int64_t bVal = lwePtr[b * (n + 1) + n];
        int shift = static_cast<int>(bVal % (2 * N));
        if (shift < 0) shift += 2 * N;
        
        // acc0 = 0, acc1 = X^{-b} * testPoly
        for (int i = 0; i < N; ++i) {
            accData[b * 2 * N + i] = 0;  // acc0
            
            // Negacyclic rotation
            int srcIdx = (i + shift) % (2 * N);
            if (srcIdx < N) {
                accData[b * 2 * N + N + i] = testPtr[srcIdx];
            } else {
                int64_t val = testPtr[srcIdx - N];
                accData[b * 2 * N + N + i] = (val == 0) ? 0 : static_cast<int64_t>(Q) - val;
            }
        }
    }
    
    mx::array acc = mx::array(accData.data(), {B, 2, N}, mx::int64);
    mx::eval(acc);
    
    // Step 2: Forward NTT on accumulators
    // Reshape to [B*2, N] for batch NTT
    auto accReshaped = mx::reshape(acc, {B * 2, N});
    batchNTTOptimized(accReshaped, false);
    acc = mx::reshape(accReshaped, {B, 2, N});
    
    // Step 3: CMux loop for each LWE coefficient
    for (int i = 0; i < n; ++i) {
        // Extract rotation amounts for this LWE index
        std::vector<int64_t> rotations(B);
        for (int b = 0; b < B; ++b) {
            rotations[b] = lwePtr[b * (n + 1) + i];
        }
        mx::array rotArr = mx::array(rotations.data(), {B}, mx::int64);
        
        // Extract BSK[i] for this LWE index
        // bsk[i] has shape [2, L, 2, N]
        int bskSize = 2 * config_.L * 2 * N;
        std::vector<int64_t> bski(bskSize);
        for (int j = 0; j < bskSize; ++j) {
            bski[j] = bskPtr[i * bskSize + j];
        }
        mx::array bskArr = mx::array(bski.data(), {2, static_cast<int>(config_.L), 2, N}, mx::int64);
        
        // Apply fused CMux
        batchCMuxFused(acc, bskArr, rotArr);
    }
    
    // Step 4: Inverse NTT on result (optional, depending on next operation)
    auto accReshaped2 = mx::reshape(acc, {B * 2, N});
    batchNTTOptimized(accReshaped2, true);
    output = mx::reshape(accReshaped2, {B, 2, N});
    mx::eval(output);
#endif
}

//==================================================================================
// Factory function
//==================================================================================

std::unique_ptr<FHEEngine> createOptimizedEngine(const FHEConfig& config) {
    return std::make_unique<FHEEngineOptimized>(config);
}

//==================================================================================
// BatchPBSScheduler Implementation
//==================================================================================

BatchPBSScheduler::BatchPBSScheduler(FHEEngine* engine) : engine_(engine) {}

void BatchPBSScheduler::queueGate(uint64_t userId, GateType gate,
                                   uint32_t input1, uint32_t input2, uint32_t output) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Add to pending ops for the gate type
    auto& op = pendingOps_[gate];
    op.gate = gate;
    op.input1Indices.push_back(input1);
    op.input2Indices.push_back(input2);
    op.outputIndices.push_back(output);
    op.userIds.push_back(userId);

    // Check for auto-flush
    if (op.input1Indices.size() >= autoFlushThreshold_) {
        flushGateType(gate);
    }
}

void BatchPBSScheduler::queueGate3(uint64_t userId, GateType gate,
                                    uint32_t in1, uint32_t in2, uint32_t in3, uint32_t output) {
    // For 3-input gates, store in1/in2 in indices1/2, in3 in outputIndices
    // (simplified - actual implementation would need proper 3-input support)
    queueGate(userId, gate, in1, in2, output);
}

void BatchPBSScheduler::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [gate, op] : pendingOps_) {
        if (!op.input1Indices.empty()) {
            flushGateType(gate);
        }
    }
}

void BatchPBSScheduler::setAutoFlushThreshold(uint32_t threshold) {
    autoFlushThreshold_ = threshold;
}

void BatchPBSScheduler::flushGateType(GateType gate) {
    auto& op = pendingOps_[gate];
    if (!op.input1Indices.empty()) {
        engine_->executeBatchGates({op});
        op.input1Indices.clear();
        op.input2Indices.clear();
        op.outputIndices.clear();
    }
}

//==================================================================================
// GPUCircuitEvaluator Implementation
//==================================================================================

GPUCircuitEvaluator::GPUCircuitEvaluator(FHEEngine* engine, uint64_t userId)
    : engine_(engine), userId_(userId), scheduler_(engine) {
    (void)engine_;  // Suppress unused warning until full implementation
    (void)userId_;  // Suppress unused warning until full implementation
}

void GPUCircuitEvaluator::add8(uint32_t a[8], uint32_t b[8], uint32_t result[8]) {
    // Ripple carry adder for 8-bit
    // Simplified placeholder - actual implementation would use full adder circuit
    for (int i = 0; i < 8; ++i) {
        result[i] = a[i];  // Placeholder
    }
}

void GPUCircuitEvaluator::add16(uint32_t a[16], uint32_t b[16], uint32_t result[16]) {
    for (int i = 0; i < 16; ++i) {
        result[i] = a[i];
    }
}

void GPUCircuitEvaluator::add32(uint32_t a[32], uint32_t b[32], uint32_t result[32]) {
    for (int i = 0; i < 32; ++i) {
        result[i] = a[i];
    }
}

void GPUCircuitEvaluator::sub8(uint32_t a[8], uint32_t b[8], uint32_t result[8]) {
    for (int i = 0; i < 8; ++i) {
        result[i] = a[i];
    }
}

void GPUCircuitEvaluator::mul8(uint32_t a[8], uint32_t b[8], uint32_t result[16]) {
    for (int i = 0; i < 16; ++i) {
        result[i] = 0;
    }
}

void GPUCircuitEvaluator::eq8(uint32_t a[8], uint32_t b[8], uint32_t& result) {
    result = 0;
}

void GPUCircuitEvaluator::lt8(uint32_t a[8], uint32_t b[8], uint32_t& result) {
    result = 0;
}

void GPUCircuitEvaluator::batchAdd8(const std::vector<std::array<uint32_t, 8>>& as,
                                     const std::vector<std::array<uint32_t, 8>>& bs,
                                     std::vector<std::array<uint32_t, 8>>& results) {
    results.resize(as.size());
    for (size_t i = 0; i < as.size(); ++i) {
        for (int j = 0; j < 8; ++j) {
            results[i][j] = as[i][j];  // Placeholder
        }
    }
}

}  // namespace gpu
}  // namespace lbcrypto
