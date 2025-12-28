//==================================================================================
// GPU TFHE CUDA - Massively Parallel TFHE for HGX H200 x8
//
// Target: 500K-2M bootstraps/sec on 8x H200 (250-1000× faster than Zama)
// Architecture: Multi-GPU with NVLink, batch-first, memory-bandwidth optimized
//==================================================================================

#ifndef GPU_TFHE_CUDA_H
#define GPU_TFHE_CUDA_H

#include <cstdint>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <functional>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <cufft.h>
#endif

namespace lbcrypto {
namespace gpu {
namespace cuda {

//==================================================================================
// Multi-GPU Configuration
//==================================================================================

struct MultiGPUConfig {
    uint32_t numGPUs = 8;                    // H200 x8
    size_t memoryPerGPU = 141ULL * 1024 * 1024 * 1024;  // 141GB HBM3e
    bool useNVLink = true;                   // NVLink mesh
    bool usePeerAccess = true;               // Direct GPU-GPU access
    
    // TFHE parameters
    uint32_t N = 1024;                       // Ring dimension
    uint32_t n = 512;                        // LWE dimension  
    uint32_t L = 4;                          // Decomposition (reduced from 7)
    uint32_t baseLog = 7;
    uint64_t Q = 1ULL << 27;
    uint64_t q = 1ULL << 15;
    
    // Batch parameters - tuned for H200
    uint32_t batchSize = 4096;               // Gates per batch (increased for H200)
    uint32_t maxUsersPerGPU = 1000;          // ~141GB / 170MB per user
    uint32_t maxTotalUsers = 8000;           // 8 GPUs × 1000 users
    
    // Memory layout
    size_t bskBytesPerUser() const {
        return n * 2 * L * 2 * N * sizeof(uint64_t);  // ~170MB
    }
    
    uint64_t mu() const { return Q / 8; }
};

//==================================================================================
// GPU Memory Handles (per-GPU)
//==================================================================================

struct GPUMemoryHandle {
    int deviceId;
    void* ptr;
    size_t bytes;
    cudaStream_t stream;
};

// Bootstrap key on single GPU
struct BootstrapKeyGPUCuda {
    int deviceId;                    // Which GPU holds this BK
    uint64_t* data;                  // Device pointer [n, 2, L, 2, N]
    uint32_t n, L, N;
    size_t bytes;
    
    // Pre-transformed NTT twiddles (per-GPU cache)
    uint64_t* nttTwiddles;
    uint64_t* nttInvTwiddles;
};

// LWE ciphertext batch on GPU
struct LWEBatchGPUCuda {
    int deviceId;
    uint64_t* a;                     // [batch, n]
    uint64_t* b;                     // [batch]
    uint32_t count;
    uint32_t capacity;
};

// RLWE ciphertext batch on GPU  
struct RLWEBatchGPUCuda {
    int deviceId;
    uint64_t* c0;                    // [batch, N]
    uint64_t* c1;                    // [batch, N]
    uint32_t count;
};

//==================================================================================
// User Session with GPU Affinity
//==================================================================================

struct UserSessionCuda {
    uint64_t userId;
    int primaryGPU;                          // GPU holding this user's keys
    
    std::shared_ptr<BootstrapKeyGPUCuda> bsk;
    
    // Ciphertext pools (can span multiple GPUs for large workloads)
    std::vector<std::shared_ptr<LWEBatchGPUCuda>> lwePools;
    
    size_t memoryUsed = 0;
    std::atomic<uint64_t> opsCompleted{0};
};

//==================================================================================
// Batch Operation Descriptor
//==================================================================================

enum class GateTypeCuda : uint8_t {
    AND, OR, XOR, NAND, NOR, XNOR, NOT, MUX,
    AND3, OR3, MAJORITY
};

struct BatchGateOpCuda {
    GateTypeCuda gate;
    
    // Inputs: can be from different users/GPUs
    std::vector<uint64_t> userIds;
    std::vector<uint32_t> input1Indices;
    std::vector<uint32_t> input2Indices;
    std::vector<uint32_t> outputIndices;
    
    // GPU routing
    std::vector<int> targetGPUs;             // Which GPU processes each op
    
    uint32_t count() const { return userIds.size(); }
};

//==================================================================================
// Multi-GPU TFHE Engine
//==================================================================================

class MultiGPUTFHEEngine {
public:
    explicit MultiGPUTFHEEngine(const MultiGPUConfig& config = MultiGPUConfig());
    ~MultiGPUTFHEEngine();
    
    // Initialization
    bool initialize();
    void shutdown();
    
    // GPU topology
    int numGPUs() const { return gpuCount_; }
    bool hasNVLink(int gpu1, int gpu2) const;
    size_t freeMemory(int gpuId) const;
    size_t totalFreeMemory() const;
    
    // User management with GPU affinity
    uint64_t createUser(int preferredGPU = -1);  // -1 = auto-assign
    void deleteUser(uint64_t userId);
    UserSessionCuda* getUser(uint64_t userId);
    int getUserGPU(uint64_t userId);
    
    // Key upload (to specific GPU)
    void uploadBootstrapKey(uint64_t userId, const std::vector<uint64_t>& bskData);
    
    // Ciphertext management
    uint32_t allocateCiphertexts(uint64_t userId, uint32_t count);
    void uploadCiphertexts(uint64_t userId, uint32_t poolIdx,
                           const std::vector<std::vector<uint64_t>>& data);
    void downloadCiphertexts(uint64_t userId, uint32_t poolIdx,
                             std::vector<std::vector<uint64_t>>& output);
    
    //==========================================================================
    // Core Batch Operations (the money makers)
    //==========================================================================
    
    // Execute batch of gates across ALL GPUs in parallel
    void executeBatchGates(const std::vector<BatchGateOpCuda>& ops);
    
    // Low-level multi-GPU kernels
    void multiGPUBatchNTT(int numGPUs, const std::vector<int>& gpuIds,
                          std::vector<uint64_t*>& polyBatches,
                          std::vector<uint32_t>& batchSizes,
                          bool inverse);
    
    void multiGPUBatchExternalProduct(
        int numGPUs, const std::vector<int>& gpuIds,
        std::vector<uint64_t*>& rlweBatches,    // [B, 2, N] per GPU
        std::vector<uint64_t*>& rgswBatches,    // [B, 2, L, 2, N] per GPU
        std::vector<uint64_t*>& outputs,
        std::vector<uint32_t>& batchSizes);
    
    void multiGPUBatchBootstrap(
        const std::vector<int>& gpuIds,
        std::vector<uint64_t*>& lweBatches,
        std::vector<uint64_t*>& bskPtrs,
        GateTypeCuda gate,
        std::vector<uint64_t*>& outputs,
        std::vector<uint32_t>& batchSizes);
    
    // Synchronization
    void syncGPU(int gpuId);
    void syncAllGPUs();
    
    // Statistics
    struct Stats {
        uint64_t totalBootstraps;
        uint64_t totalGates;
        double avgBootstrapTimeUs;
        double throughputGatesPerSec;
        size_t totalMemoryUsed;
        std::vector<size_t> memoryPerGPU;
    };
    Stats getStats() const;
    
private:
    MultiGPUConfig config_;
    int gpuCount_ = 0;
    
    // Per-GPU resources
    struct GPUContext {
        cudaStream_t computeStream;
        cudaStream_t copyStream;
        cufftHandle fftPlan;
        
        // Pre-allocated scratch buffers
        uint64_t* scratchBuffer;
        size_t scratchSize;
        
        // NTT twiddles (shared across all users on this GPU)
        uint64_t* nttTwiddles;
        uint64_t* nttInvTwiddles;
        
        // Test polynomials for each gate type
        uint64_t* testPolynomials;  // [numGates, N]
        
        // Memory tracking
        size_t memoryUsed = 0;
        size_t memoryAvailable;
    };
    std::vector<GPUContext> gpuContexts_;
    
    // User management
    std::unordered_map<uint64_t, std::unique_ptr<UserSessionCuda>> users_;
    std::mutex usersMutex_;
    std::atomic<uint64_t> nextUserId_{1};
    
    // GPU assignment
    std::vector<uint32_t> usersPerGPU_;  // Count of users on each GPU
    int findBestGPU();                    // Load balancing
    
    // NVLink topology
    std::vector<std::vector<bool>> nvlinkMatrix_;
    void detectNVLinkTopology();
    
    // Initialization helpers
    void initializeGPU(int gpuId);
    void initializeNTTTwiddles(int gpuId);
    void initializeTestPolynomials(int gpuId);
    void enablePeerAccess();
    
    // Work distribution
    struct WorkPartition {
        std::vector<std::vector<uint32_t>> opsPerGPU;  // Indices into original batch
        std::vector<uint32_t> countPerGPU;
    };
    WorkPartition partitionWork(const std::vector<BatchGateOpCuda>& ops);
    
    // Stats tracking
    mutable std::mutex statsMutex_;
    Stats stats_;
};

//==================================================================================
// Batch PBS Scheduler for Multi-GPU
//==================================================================================

class MultiGPUBatchScheduler {
public:
    explicit MultiGPUBatchScheduler(MultiGPUTFHEEngine* engine);
    
    // Queue operations (automatically routes to correct GPU)
    void queueGate(uint64_t userId, GateTypeCuda gate,
                   uint32_t input1, uint32_t input2, uint32_t output);
    
    // Execute all queued operations with optimal GPU parallelism
    void flush();
    
    // Configuration
    void setBatchSize(uint32_t size) { batchSize_ = size; }
    void setAutoFlush(bool enabled) { autoFlush_ = enabled; }
    
private:
    MultiGPUTFHEEngine* engine_;
    
    // Pending operations grouped by gate type
    std::unordered_map<GateTypeCuda, BatchGateOpCuda> pending_;
    std::mutex mutex_;
    
    uint32_t batchSize_ = 4096;
    bool autoFlush_ = true;
    
    void flushIfNeeded();
};

//==================================================================================
// Performance Estimation
//==================================================================================

struct PerformanceEstimate {
    // Hardware
    int numGPUs;
    size_t totalMemoryGB;
    double aggregateBandwidthTBps;
    
    // Capacity
    uint32_t maxConcurrentUsers;
    uint64_t maxCiphertextsTotal;
    
    // Throughput
    double peakBootstrapsPerSec;
    double peakGatesPerSec;
    double estimatedLatencyMs;  // For single gate
    
    // Comparison
    double speedupVsZamaSingleGPU;
    double speedupVsZamaCPU;
};

PerformanceEstimate estimatePerformance(const MultiGPUConfig& config);

//==================================================================================
// Convenience Functions
//==================================================================================

// Quick setup for common configurations
MultiGPUTFHEEngine* createHGXH200x8Engine();   // 8x H200 optimized
MultiGPUTFHEEngine* createDGXH100x8Engine();   // 8x H100 optimized  
MultiGPUTFHEEngine* createSingleGPUEngine();   // Single GPU mode

}  // namespace cuda
}  // namespace gpu
}  // namespace lbcrypto

#endif  // GPU_TFHE_CUDA_H
