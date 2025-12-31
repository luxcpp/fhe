//==================================================================================
// GPU FHE - Massively Parallel FHE for Large-Scale Deployments
//
// Architecture: Everything on GPU, zero CPU roundtrips during computation
// Target: 1000+ concurrent users, 100GB+ GPU memory
// Design: Batch-first, fused kernels, coalesced memory access
//==================================================================================

#ifndef GPU_FHE_H
#define GPU_FHE_H

#include <cstdint>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <atomic>

#ifdef WITH_MLX
#include <mlx/mlx.h>
namespace mx = mlx::core;
#endif

namespace lbcrypto {
namespace gpu {

// Forward declarations
class NTTEngine;

namespace metal {
class FHEMetalDispatcher;
}

namespace gpu {
class NTTOptimal;
class BlindRotate;
class KeySwitch;
}

//==================================================================================
// Configuration
//==================================================================================

struct FHEConfig {
    // Ring parameters
    uint32_t N = 1024;           // Ring dimension (blind rotation)
    uint32_t n = 512;            // LWE dimension
    uint32_t L = 4;              // Decomposition digits (reduced from 7!)
    uint32_t baseLog = 7;        // Log2 of decomposition base
    // NTT-friendly prime: Q ≡ 1 (mod 2N) required for primitive root of unity
    // 268441601 = 131075 * 2048 + 1 = 0x10001801
    uint64_t Q = 268441601ULL;   // Ring modulus (NTT-friendly prime ~2^28)
    uint64_t q = 1ULL << 15;     // LWE modulus
    
    // Batch parameters
    uint32_t maxUsers = 10000;           // Max concurrent users
    uint32_t maxCiphertextsPerUser = 1000;  // Max ciphertexts per user
    uint32_t batchSize = 256;            // Operations per batch
    
    // Memory budget (bytes)
    size_t gpuMemoryBudget = 100ULL * 1024 * 1024 * 1024;  // 100GB default
    
    // Encoding
    uint64_t mu() const { return Q / 8; }  // ±μ encoding
};

//==================================================================================
// GPU Memory Layout (Structure of Arrays for coalescing)
//==================================================================================

// LWE Ciphertext: (a[0..n-1], b) where b = <a,s> + m + e
struct LWECiphertextGPU {
    // For batch of B ciphertexts:
    // a: [B, n] - mask vectors (SoA: all a[0]s, then all a[1]s, etc.)
    // b: [B]    - body values
#ifdef WITH_MLX
    std::shared_ptr<mx::array> a;  // Shape: [batch, n]
    std::shared_ptr<mx::array> b;  // Shape: [batch]
#endif
    uint32_t count = 0;  // Number of valid ciphertexts
};

// RLWE Ciphertext: (a(X), b(X)) polynomials in NTT domain
struct RLWECiphertextGPU {
    // For batch of B ciphertexts:
    // c0: [B, N] - first polynomial (NTT domain)
    // c1: [B, N] - second polynomial (NTT domain)
#ifdef WITH_MLX
    std::shared_ptr<mx::array> c0;  // Shape: [batch, N]
    std::shared_ptr<mx::array> c1;  // Shape: [batch, N]
#endif
    uint32_t count = 0;
};

// RGSW Ciphertext (Bootstrap Key Entry)
// For one BK[i]: encrypts X^{s[i]}
// Structure: [2, L, 2, N] in NTT domain
struct RSGWCiphertextGPU {
#ifdef WITH_MLX
    std::shared_ptr<mx::array> data;  // Shape: [2, L, 2, N] - fully in NTT domain
#endif
};

// Full Bootstrap Key for one user
struct BootstrapKeyGPU {
    // BK[i] = RGSW(X^{s[i]}) for i = 0..n-1
    // Layout: [n, 2, L, 2, N] - all in NTT domain, all on GPU
#ifdef WITH_MLX
    std::shared_ptr<mx::array> data;  // Shape: [n, 2, L, 2, N]
#endif
    uint32_t n = 0;
    uint32_t L = 0;
    uint32_t N = 0;
    
    size_t memoryBytes() const {
        return n * 2 * L * 2 * N * sizeof(uint64_t);
    }
};

// Key Switching Key (for LWE dimension reduction after blind rotation)
struct KeySwitchKeyGPU {
#ifdef WITH_MLX
    std::shared_ptr<mx::array> data;  // Shape: [N, L_ks, n]
#endif
};

//==================================================================================
// User Session - Isolated context per tenant
//==================================================================================

struct UserSession {
    uint64_t userId;
    
    // Keys (on GPU)
    std::shared_ptr<BootstrapKeyGPU> bsk;
    std::shared_ptr<KeySwitchKeyGPU> ksk;
    
    // Ciphertext pools (on GPU)
    std::vector<LWECiphertextGPU> lwePools;
    std::vector<RLWECiphertextGPU> rlwePools;
    
    // Operation queue
    std::atomic<uint32_t> pendingOps{0};
    
    // Memory tracking
    size_t memoryUsed = 0;
};

//==================================================================================
// Operation Batch - Grouped operations for parallel execution
//==================================================================================

enum class GateType : uint8_t {
    AND, OR, XOR, NAND, NOR, XNOR, NOT, MUX,
    AND3, OR3, MAJORITY
};

struct BatchedGateOp {
    GateType gate;
    
    // Input ciphertext indices (within user's pool)
    std::vector<uint32_t> input1Indices;
    std::vector<uint32_t> input2Indices;
    std::vector<uint32_t> input3Indices;  // For 3-input gates
    
    // Output indices
    std::vector<uint32_t> outputIndices;
    
    // User IDs (for multi-user batching)
    std::vector<uint64_t> userIds;
    
    uint32_t count() const { return input1Indices.size(); }
};

//==================================================================================
// GPU FHE Engine - Main interface
//==================================================================================

class FHEEngine {
public:
    explicit FHEEngine(const FHEConfig& config = FHEConfig());
    virtual ~FHEEngine();
    
    // Initialization
    virtual bool initialize();
    void shutdown();
    
    // User management
    uint64_t createUser();
    void deleteUser(uint64_t userId);
    UserSession* getUser(uint64_t userId);
    
    // Key generation (done once per user, stored on GPU)
    void generateKeys(uint64_t userId);
    void uploadBootstrapKey(uint64_t userId, const std::vector<uint64_t>& bskData);
    
    // Ciphertext management
    uint32_t allocateCiphertexts(uint64_t userId, uint32_t count);
    void uploadCiphertexts(uint64_t userId, uint32_t startIdx, 
                           const std::vector<std::vector<uint64_t>>& data);
    void downloadCiphertexts(uint64_t userId, uint32_t startIdx, uint32_t count,
                             std::vector<std::vector<uint64_t>>& output);
    
    // Batch gate operations (the main workhorse)
    void executeBatchGates(const std::vector<BatchedGateOp>& ops);
    
    // Synchronization
    void sync();
    
    // Stats
    size_t totalGPUMemoryUsed() const;
    size_t availableGPUMemory() const;
    uint32_t activeUsers() const;
    double avgOperationsPerSecond() const;

#ifdef WITH_MLX
    // Core GPU kernels (public for testing/benchmarking)
    void batchNTT(mx::array& polys, bool inverse);
    void batchExternalProduct(const mx::array& rlweBatch,    // [B, 2, N]
                              const mx::array& rgswBatch,    // [B, 2, L, 2, N]
                              mx::array& output);            // [B, 2, N]
    void batchBlindRotate(const mx::array& lweBatch,         // [B, n+1]
                          const mx::array& bskBatch,         // [B, n, 2, L, 2, N]
                          const mx::array& testPoly,         // [B, N] or [N]
                          mx::array& output);                // [B, 2, N]
    void batchKeySwitch(const mx::array& rlweBatch,          // [B, 2, N]
                        const mx::array& kskBatch,           // [B, N, L_ks, n]
                        mx::array& output);                  // [B, n+1]
    void batchBootstrap(const mx::array& lweBatch,           // [B, n+1]
                        GateType gate,
                        const mx::array& bskBatch,
                        const mx::array& kskBatch,
                        mx::array& output);
#endif

protected:
    const FHEConfig& config() const { return config_; }
    FHEConfig config_;
    
    // User sessions
    std::unordered_map<uint64_t, std::unique_ptr<UserSession>> users_;
    std::mutex usersMutex_;
    std::atomic<uint64_t> nextUserId_{1};
    
    // Precomputed data on GPU
#ifdef WITH_MLX
    std::shared_ptr<mx::array> twiddleFactors_;      // NTT twiddle factors [N]
    std::shared_ptr<mx::array> invTwiddleFactors_;   // Inverse NTT twiddles [N]
    std::shared_ptr<mx::array> testPolynomials_;     // Gate test polynomials [numGates, N]
    
    // NTT engine for Montgomery-optimized transforms
    std::unique_ptr<NTTEngine> nttEngine_;

    // Metal GPU dispatcher (GPU-first, CPU-fallback)
    std::unique_ptr<metal::FHEMetalDispatcher> metalDispatcher_;
    bool useGpu_ = false;
#endif

    // Internal methods
    void initializeNTTTwiddles();
    void initializeTestPolynomials();
};

//==================================================================================
// Batch PBS Scheduler - Optimizes operation ordering
//==================================================================================

class BatchPBSScheduler {
public:
    explicit BatchPBSScheduler(FHEEngine* engine);
    
    // Queue operations from multiple users
    void queueGate(uint64_t userId, GateType gate,
                   uint32_t input1, uint32_t input2, uint32_t output);
    void queueGate3(uint64_t userId, GateType gate,
                    uint32_t in1, uint32_t in2, uint32_t in3, uint32_t output);
    
    // Execute all queued operations
    void flush();
    
    // Auto-flush when batch is full
    void setAutoFlushThreshold(uint32_t threshold);
    
private:
    FHEEngine* engine_;
    std::unordered_map<GateType, BatchedGateOp> pendingOps_;
    uint32_t autoFlushThreshold_ = 256;
    std::mutex mutex_;
    
    void flushGateType(GateType gate);
};

//==================================================================================
// High-Level Circuit Evaluator
//==================================================================================

class GPUCircuitEvaluator {
public:
    GPUCircuitEvaluator(FHEEngine* engine, uint64_t userId);
    
    // Integer operations (using batch gates internally)
    void add8(uint32_t a[8], uint32_t b[8], uint32_t result[8]);
    void add16(uint32_t a[16], uint32_t b[16], uint32_t result[16]);
    void add32(uint32_t a[32], uint32_t b[32], uint32_t result[32]);
    
    void sub8(uint32_t a[8], uint32_t b[8], uint32_t result[8]);
    void mul8(uint32_t a[8], uint32_t b[8], uint32_t result[16]);
    
    void eq8(uint32_t a[8], uint32_t b[8], uint32_t& result);
    void lt8(uint32_t a[8], uint32_t b[8], uint32_t& result);
    
    // Batched integer operations (process many integers at once)
    void batchAdd8(const std::vector<std::array<uint32_t, 8>>& as,
                   const std::vector<std::array<uint32_t, 8>>& bs,
                   std::vector<std::array<uint32_t, 8>>& results);
    
private:
    [[maybe_unused]] FHEEngine* engine_;
    [[maybe_unused]] uint64_t userId_;
    BatchPBSScheduler scheduler_;
};

//==================================================================================
// Factory Function
//==================================================================================

/**
 * @brief Create an optimized FHE engine for the current platform
 * @param config FHE configuration parameters
 * @return Unique pointer to FHE engine (MLX backend on Apple Silicon, CPU fallback otherwise)
 */
std::unique_ptr<FHEEngine> createOptimizedEngine(const FHEConfig& config);

}  // namespace gpu
}  // namespace lbcrypto

#endif  // GPU_FHE_H
