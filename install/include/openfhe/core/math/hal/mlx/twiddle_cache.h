// =============================================================================
// Twiddle Hotset Cache for GPU-Accelerated FHE
// =============================================================================
//
// Intelligent caching of frequently-used NTT twiddle factors with LRU eviction.
//
// Key insight: NTT stages have vastly different twiddle requirements:
//   - Stage 0: 1 twiddle (always 1)
//   - Stage 1: 2 twiddles
//   - Stage 2: 4 twiddles
//   ...
//   - Stage 9: 512 twiddles (for N=1024)
//
// Early stages (0-5) use <= 32 twiddles total, fitting entirely in threadgroup
// memory. This allows complete elimination of global memory traffic for small
// NTT dimensions.
//
// For multi-modulus RNS, we maintain separate caches per modulus with LRU
// eviction when threadgroup memory pressure exceeds capacity.
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: BSD-2-Clause
// =============================================================================

#ifndef LBCRYPTO_MATH_HAL_MLX_TWIDDLE_CACHE_H
#define LBCRYPTO_MATH_HAL_MLX_TWIDDLE_CACHE_H

#include <cstdint>
#include <vector>
#include <memory>
#include <array>
#include <mutex>
#include <atomic>
#include <chrono>
#include <optional>

#ifdef WITH_MLX
#include <mlx/mlx.h>
namespace mx = mlx::core;
#endif

namespace lbcrypto {
namespace mlx_backend {

// =============================================================================
// Constants
// =============================================================================

/// Maximum number of RNS primes supported
static constexpr size_t MAX_RNS_PRIMES = 16;

/// Number of twiddles stored in constant memory (first-level cache)
static constexpr size_t FIRST_LEVEL_TWIDDLE_COUNT = 8;

/// Apple M3 threadgroup memory: 32KB
static constexpr size_t M3_THREADGROUP_BYTES = 32 * 1024;

/// Maximum twiddles in threadgroup memory: 32KB / 8 bytes = 4096
static constexpr size_t MAX_THREADGROUP_TWIDDLES = M3_THREADGROUP_BYTES / sizeof(uint64_t);

/// Cache line size for alignment (64 bytes = 8 uint64_t)
static constexpr size_t CACHE_LINE_BYTES = 64;
static constexpr size_t CACHE_LINE_ELEMENTS = CACHE_LINE_BYTES / sizeof(uint64_t);

// =============================================================================
// Prime Constants for Modular Arithmetic
// =============================================================================

/// Pre-computed constants for a single prime modulus
struct PrimeConstants {
    uint64_t q;           ///< The prime modulus
    uint64_t q_inv;       ///< -q^(-1) mod 2^64 (Montgomery constant)
    uint64_t mu_hi;       ///< Barrett constant high 64 bits: floor(2^128/q) >> 64
    uint64_t mu_lo;       ///< Barrett constant low 64 bits
    uint64_t r_squared;   ///< R^2 mod q where R=2^64 (Montgomery domain entry)
    uint64_t root;        ///< Primitive N-th root of unity mod q
    uint64_t root_inv;    ///< Inverse root: root^(-1) mod q
    uint64_t n_inv;       ///< N^(-1) mod q for INTT scaling

    PrimeConstants() = default;
    PrimeConstants(uint64_t prime, uint64_t ring_dim);
};

// =============================================================================
// Stage-Specific Twiddle Cache
// =============================================================================

/// Twiddles for a single NTT stage, optimized for threadgroup loading
struct StageTwiddleCache {
    uint32_t stageIndex;      ///< Stage number (0 to log_N - 1)
    uint32_t twiddleCount;    ///< Number of unique twiddles for this stage
    uint32_t stride;          ///< Access stride in the NTT butterfly
    std::vector<uint64_t> twiddles;  ///< Pre-computed twiddle values

    StageTwiddleCache() = default;
    StageTwiddleCache(uint32_t stage, uint32_t N, uint64_t q, uint64_t root);
};

// =============================================================================
// Twiddle Memory Layout
// =============================================================================

/// Memory layout options for multi-prime twiddle storage
enum class TwiddleLayout {
    /// [prime][twiddle]: Good for single-prime access
    PRIME_MAJOR,
    /// [twiddle][prime]: Good for RNS parallel access
    TWIDDLE_MAJOR
};

// =============================================================================
// Constant Memory Cache (L1 - Fastest)
// =============================================================================

/// Data stored in GPU constant memory for instant access
struct ConstantMemoryCache {
    uint32_t numPrimes;
    uint32_t ringDim;
    uint32_t _padding[2];

    PrimeConstants primes[MAX_RNS_PRIMES];

    /// First-level twiddles per prime (covers stages 0-3)
    uint64_t firstLevelTwiddles[MAX_RNS_PRIMES][FIRST_LEVEL_TWIDDLE_COUNT];
    uint64_t firstLevelInvTwiddles[MAX_RNS_PRIMES][FIRST_LEVEL_TWIDDLE_COUNT];
};

// =============================================================================
// Device Memory Twiddle Table (L3 - Complete)
// =============================================================================

/// Complete twiddle tables in device memory
struct DeviceTwiddleTable {
    uint32_t numPrimes;
    uint32_t ringDim;
    TwiddleLayout layout;

    std::vector<uint64_t> forwardTwiddles;
    std::vector<uint64_t> inverseTwiddles;
    std::vector<uint64_t> forwardTwiddlesBitRev;
    std::vector<uint64_t> inverseTwiddlesBitRev;

    size_t sizeBytes() const {
        return (forwardTwiddles.size() + inverseTwiddles.size() +
                forwardTwiddlesBitRev.size() + inverseTwiddlesBitRev.size()) * sizeof(uint64_t);
    }
};

// =============================================================================
// Cache Access Statistics
// =============================================================================

struct CacheStats {
    mutable std::atomic<uint64_t> totalAccesses{0};
    mutable std::atomic<uint64_t> constantHits{0};
    mutable std::atomic<uint64_t> threadgroupHits{0};
    mutable std::atomic<uint64_t> deviceAccesses{0};
    mutable std::atomic<uint64_t> cacheMisses{0};

    void reset() {
        totalAccesses = 0;
        constantHits = 0;
        threadgroupHits = 0;
        deviceAccesses = 0;
        cacheMisses = 0;
    }

    double hitRate() const {
        uint64_t total = totalAccesses.load();
        if (total == 0) return 0.0;
        return 1.0 - (double)cacheMisses.load() / total;
    }

    double constantHitRate() const {
        uint64_t total = totalAccesses.load();
        if (total == 0) return 0.0;
        return (double)constantHits.load() / total;
    }
};

// =============================================================================
// LRU Entry for Multi-Modulus Cache Management
// =============================================================================

/// LRU cache entry tracking access recency
struct LRUEntry {
    uint32_t primeIndex;       ///< Which prime modulus
    uint32_t stageIndex;       ///< Which NTT stage
    uint64_t lastAccessTime;   ///< Monotonic timestamp of last access
    bool valid;                ///< Entry is loaded and valid

    LRUEntry() : primeIndex(0), stageIndex(0), lastAccessTime(0), valid(false) {}
};

/// LRU cache manager for threadgroup memory
class LRUCacheManager {
public:
    explicit LRUCacheManager(size_t capacity);

    /// Record an access to (prime, stage), returns true if already cached
    bool access(uint32_t prime, uint32_t stage);

    /// Get the entry to evict when cache is full
    std::pair<uint32_t, uint32_t> evict();

    /// Check if (prime, stage) is currently cached
    bool contains(uint32_t prime, uint32_t stage) const;

    /// Clear all entries
    void clear();

    /// Current number of cached entries
    size_t size() const { return currentSize_; }

    /// Maximum capacity
    size_t capacity() const { return capacity_; }

private:
    size_t capacity_;
    size_t currentSize_;
    std::vector<LRUEntry> entries_;
    std::atomic<uint64_t> accessCounter_;

    size_t findSlot(uint32_t prime, uint32_t stage) const;
    size_t findVictim() const;
};

// =============================================================================
// Main Twiddle Cache Class
// =============================================================================

/// Hierarchical twiddle cache with LRU management for GPU FHE
class TwiddleCache {
public:
    /// Construct cache for given ring dimension and primes
    TwiddleCache(uint32_t ringDim, const std::vector<uint64_t>& primes);

    ~TwiddleCache();

    // Move-only (no copy due to GPU resources)
    TwiddleCache(const TwiddleCache&) = delete;
    TwiddleCache& operator=(const TwiddleCache&) = delete;
    TwiddleCache(TwiddleCache&&) noexcept;
    TwiddleCache& operator=(TwiddleCache&&) noexcept;

    // =========================================================================
    // Precomputation
    // =========================================================================

    /// Precompute all twiddle factors and constants
    void precompute();

    /// Upload cache to GPU and warm memory hierarchy
    void warmCache();

    /// Verify cache is performing as expected
    bool verifyCachePerformance();

    // =========================================================================
    // Twiddle Access
    // =========================================================================

    /// Get forward twiddle for prime at index
    uint64_t getForwardTwiddle(uint32_t primeIdx, uint32_t twiddleIdx) const;

    /// Get inverse twiddle for prime at index
    uint64_t getInverseTwiddle(uint32_t primeIdx, uint32_t twiddleIdx) const;

    /// Batch access for vectorized operations
    void getForwardTwiddleBatch(uint32_t primeIdx, uint32_t startIdx,
                                 uint32_t count, uint64_t* output) const;
    void getInverseTwiddleBatch(uint32_t primeIdx, uint32_t startIdx,
                                 uint32_t count, uint64_t* output) const;

    /// Get prime constants
    const PrimeConstants& getPrimeConstants(uint32_t primeIdx) const;

    /// Get stage-specific twiddles for GPU upload
#ifdef WITH_MLX
    mx::array getStageTwiddles(uint32_t stageIdx, uint32_t primeIdx) const;
#endif

    // =========================================================================
    // Cache Configuration
    // =========================================================================

    /// Set memory layout for device twiddles
    void setLayout(TwiddleLayout layout);

    /// Get current layout
    TwiddleLayout getLayout() const { return deviceTable_.layout; }

    // =========================================================================
    // Statistics and Analysis
    // =========================================================================

    /// Get cache statistics
    const CacheStats& stats() const { return stats_; }

    /// Reset statistics
    void resetStats() { stats_.reset(); }

    /// Estimate hit rate for given parameters
    static double estimateHitRate(uint32_t ringDim, uint32_t batchSize);

    /// Estimate memory bandwidth savings
    static double estimateBandwidthSavings(uint32_t ringDim, uint32_t batchSize);

    // =========================================================================
    // Memory Usage
    // =========================================================================

    /// Total memory used by cache
    size_t totalMemoryBytes() const;

    /// Memory used in constant tier
    size_t constantMemoryBytes() const { return sizeof(ConstantMemoryCache); }

    /// Memory used in device tier
    size_t deviceMemoryBytes() const { return deviceTable_.sizeBytes(); }

    // =========================================================================
    // Accessors
    // =========================================================================

    uint32_t ringDim() const { return ringDim_; }
    uint32_t logRingDim() const { return logRingDim_; }
    size_t numPrimes() const { return primes_.size(); }
    const std::vector<uint64_t>& primes() const { return primes_; }
    bool isPrecomputed() const { return isPrecomputed_; }
    bool isCacheWarm() const { return isCacheWarm_; }

private:
    // Configuration
    uint32_t ringDim_;
    uint32_t logRingDim_;
    std::vector<uint64_t> primes_;

    // Cache tiers
    ConstantMemoryCache constantCache_;
    std::vector<std::vector<StageTwiddleCache>> stageCaches_;  // [prime][stage]
    DeviceTwiddleTable deviceTable_;

    // LRU management for multi-modulus
    std::unique_ptr<LRUCacheManager> lruManager_;

    // GPU resources
#ifdef WITH_MLX
    std::optional<mx::array> constantBuffer_;
    std::optional<mx::array> deviceForwardTwiddles_;
    std::optional<mx::array> deviceInverseTwiddles_;
    std::vector<std::vector<std::optional<mx::array>>> stageBuffers_;  // [prime][stage]
#endif

    // State
    bool isPrecomputed_ = false;
    bool isCacheWarm_ = false;
    mutable CacheStats stats_;
    mutable std::mutex cacheMutex_;

    // Precomputation helpers
    uint64_t computePrimitiveRoot(uint64_t N, uint64_t q) const;
    void computeBarrettConstant(uint64_t q, uint64_t& mu_hi, uint64_t& mu_lo) const;
    uint64_t computeMontgomeryConstant(uint64_t q) const;
    uint64_t computeRSquared(uint64_t q) const;
    void computeTwiddleTable(uint64_t q, uint64_t root, std::vector<uint64_t>& twiddles) const;
    void bitReversePermute(std::vector<uint64_t>& arr) const;
    uint32_t bitReverse(uint32_t x, uint32_t bits) const;
    uint64_t modInverse(uint64_t a, uint64_t m) const;
    uint64_t modPow(uint64_t base, uint64_t exp, uint64_t m) const;

    // Internal initialization
    void computeFirstLevelTwiddles(uint32_t primeIdx);
    void precomputeStageCaches(uint32_t primeIdx);
};

// =============================================================================
// Factory Functions
// =============================================================================

/// Create cache optimized for TFHE (single prime)
std::unique_ptr<TwiddleCache> createTFHECache(uint32_t N, uint64_t Q);

/// Create cache for RNS-based CKKS/BGV (multiple primes)
std::unique_ptr<TwiddleCache> createRNSCache(uint32_t N, const std::vector<uint64_t>& primes);

// =============================================================================
// Utility Functions
// =============================================================================

/// Check if prime is NTT-friendly for given dimension
bool isNTTFriendly(uint64_t q, uint32_t N);

/// Generate NTT-friendly primes
std::vector<uint64_t> generateNTTFriendlyPrimes(uint32_t N, uint32_t bitWidth, uint32_t count);

// =============================================================================
// Hotset Configuration
// =============================================================================

/// Configuration for hotset caching behavior
struct HotsetConfig {
    /// Maximum stages to keep in constant memory (default: 4, covers stages 0-3)
    uint32_t constantMemoryStages = 4;

    /// Maximum twiddles per stage in threadgroup memory
    uint32_t maxThreadgroupTwiddles = MAX_THREADGROUP_TWIDDLES;

    /// Enable prefetch hints for next stage during compute
    bool enablePrefetch = true;

    /// Enable LRU eviction for multi-modulus pressure
    bool enableLRU = true;

    /// LRU cache capacity (entries = prime * stage pairs)
    size_t lruCapacity = 64;

    /// Compute optimal configuration for given parameters
    static HotsetConfig optimal(uint32_t ringDim, uint32_t numPrimes);
};

/// Compute memory requirements for hotset caching
struct HotsetMemoryEstimate {
    size_t constantBytes;      ///< Constant memory usage
    size_t threadgroupBytes;   ///< Peak threadgroup memory per threadgroup
    size_t deviceBytes;        ///< Device memory for complete tables

    double bandwidthReduction; ///< Estimated memory bandwidth reduction (0-1)
    double latencyReduction;   ///< Estimated latency reduction (0-1)

    static HotsetMemoryEstimate compute(uint32_t ringDim, uint32_t numPrimes);
};

}  // namespace mlx_backend
}  // namespace lbcrypto

#endif  // LBCRYPTO_MATH_HAL_MLX_TWIDDLE_CACHE_H
