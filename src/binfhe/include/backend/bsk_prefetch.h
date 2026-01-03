// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// Speculative Bootstrapping Key Prefetcher
//
// Predicts BSK tile access sequence from LWE ciphertext bits and prefetches
// tiles while previous operations execute. Hides multi-GB key transfer latency
// behind computation.

#ifndef BACKEND_BSK_PREFETCH_H
#define BACKEND_BSK_PREFETCH_H

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lux::fhe {
namespace backend {

// ============================================================================
// BSK Layout Description
// ============================================================================

/**
 * @brief Describes the memory layout of a bootstrapping key
 *
 * The BSK is organized as: key[dim1][dim2][dim3]
 * where:
 *   dim1 = base (typically 1 for binary secrets)
 *   dim2 = sign (0 for positive, 1 for negative secret key bit)
 *   dim3 = LWE dimension index
 *
 * Each element is an RGSW ciphertext.
 */
struct BSKLayout {
    // Dimensions
    uint32_t n;              // LWE dimension (typically 512-1024)
    uint32_t digits_g;       // Gadget decomposition digits (2-4)
    uint32_t ring_dim;       // Ring dimension N (1024-4096)
    uint32_t rlwe_dim;       // RLWE dimension k (typically 1)

    // Tile parameters
    uint32_t tile_n;         // Number of LWE indices per tile
    size_t tile_size;        // Bytes per tile
    uint32_t num_tiles;      // Total tiles = ceil(n / tile_n)

    // Memory pointers
    void* base_ptr;          // Start of BSK in slow memory (host/HBM)
    void* cache_ptr;         // Fast memory cache (L2/shared)
    size_t total_size;       // Total BSK size in bytes

    // Element size computation
    size_t ElementSize() const {
        // Size of one RGSW ciphertext
        // RGSW has (digits_g - 1) * 2 rows, each row is 2 polynomials of ring_dim elements
        size_t row_size = ring_dim * sizeof(uint64_t) * 2;  // a,b polynomials
        size_t rgsw_size = row_size * (digits_g - 1) * 2;   // All rows
        return rgsw_size;
    }

    // Size of BSK data for one LWE index
    size_t PerIndexSize() const {
        // Both ek[0][0][i] and ek[0][1][i]
        return ElementSize() * 2;
    }

    // Compute tile index from LWE index
    uint32_t TileIndex(uint32_t lwe_idx) const {
        return lwe_idx / tile_n;
    }

    // Compute offset within tile
    uint32_t TileOffset(uint32_t lwe_idx) const {
        return lwe_idx % tile_n;
    }

    // Compute pointer to start of tile
    void* TilePtr(uint32_t tile_idx) const {
        size_t offset = tile_idx * tile_size;
        return static_cast<uint8_t*>(base_ptr) + offset;
    }
};

// ============================================================================
// Device Memory Profile
// ============================================================================

/**
 * @brief Describes device memory hierarchy characteristics
 */
struct DeviceMemoryProfile {
    size_t l1_cache_size;        // Per-SM L1 cache (bytes)
    size_t l2_cache_size;        // Shared L2 cache (bytes)
    size_t shared_memory_size;   // Per-SM shared memory (bytes)
    size_t register_file_size;   // Per-SM registers (bytes)

    double hbm_bandwidth;        // HBM bandwidth (GB/s)
    double l2_bandwidth;         // L2 bandwidth (GB/s)
    double compute_throughput;   // Compute throughput (GFLOPS)

    double memory_latency_us;    // HBM access latency (microseconds)

    // Compute optimal tile size for this device
    uint32_t OptimalTileN(const BSKLayout& layout) const;

    // Default profile for Apple M1 Max
    static DeviceMemoryProfile M1Max();

    // Default profile for NVIDIA A100
    static DeviceMemoryProfile A100();

    // Default profile for CPU with large L3
    static DeviceMemoryProfile CPU();
};

// ============================================================================
// Tile Access Prediction
// ============================================================================

/**
 * @brief Predicted tile access during blind rotation
 */
struct TileAccessPrediction {
    uint32_t tile_index;     // Which tile will be accessed
    uint32_t lwe_index;      // First LWE index that triggers access
    uint32_t priority;       // Lower = accessed sooner

    bool operator<(const TileAccessPrediction& other) const {
        return priority > other.priority;  // Min-heap by priority
    }
};

/**
 * @brief Predicts BSK tile access sequence from LWE a-vector
 *
 * The blind rotation accesses ek[0][0][i] and ek[0][1][i] for each
 * LWE coefficient i. By examining the a-vector before blind rotation,
 * we can predict the entire tile access sequence.
 *
 * @param a_vector Pointer to LWE a-vector coefficients
 * @param n Length of a-vector
 * @param layout BSK memory layout
 * @return Vector of tile access predictions, sorted by access order
 */
std::vector<TileAccessPrediction> PredictAccessSequence(
    const uint64_t* a_vector,
    uint32_t n,
    const BSKLayout& layout
);

// ============================================================================
// Prefetch Request
// ============================================================================

/**
 * @brief Request to prefetch a BSK tile
 */
struct PrefetchRequest {
    uint32_t tile_index;
    uint32_t priority;
    void* source_ptr;        // Source in slow memory
    void* dest_ptr;          // Destination in fast memory
    size_t size;
    std::atomic<bool> completed{false};
    std::atomic<bool> in_flight{false};

    PrefetchRequest() = default;
    PrefetchRequest(const PrefetchRequest& other)
        : tile_index(other.tile_index)
        , priority(other.priority)
        , source_ptr(other.source_ptr)
        , dest_ptr(other.dest_ptr)
        , size(other.size)
        , completed(other.completed.load())
        , in_flight(other.in_flight.load()) {}

    PrefetchRequest& operator=(const PrefetchRequest& other) {
        tile_index = other.tile_index;
        priority = other.priority;
        source_ptr = other.source_ptr;
        dest_ptr = other.dest_ptr;
        size = other.size;
        completed = other.completed.load();
        in_flight = other.in_flight.load();
        return *this;
    }

    bool operator<(const PrefetchRequest& other) const {
        return priority > other.priority;  // Min-heap
    }
};

// ============================================================================
// Prefetch Queue
// ============================================================================

/**
 * @brief Priority queue for prefetch requests
 */
class PrefetchQueue {
public:
    PrefetchQueue() = default;

    /**
     * @brief Enqueue predictions as prefetch requests
     */
    void EnqueuePredictions(
        const std::vector<TileAccessPrediction>& predictions,
        const BSKLayout& layout
    );

    /**
     * @brief Pop highest-priority request
     */
    std::optional<PrefetchRequest> PopHighestPriority();

    /**
     * @brief Check if tile is already queued
     */
    bool IsQueued(uint32_t tile_index) const;

    /**
     * @brief Check if queue is empty
     */
    bool Empty() const;

    /**
     * @brief Clear all pending requests
     */
    void Clear();

    /**
     * @brief Get number of pending requests
     */
    size_t Size() const;

private:
    std::priority_queue<PrefetchRequest> queue_;
    std::unordered_set<uint32_t> queued_tiles_;
    mutable std::mutex mutex_;

    // Allocate cache slot for tile
    void* AllocateCacheSlot(uint32_t tile_index);
};

// ============================================================================
// Double-Buffered Cache
// ============================================================================

/**
 * @brief Buffer slot states
 */
enum class BufferState {
    EMPTY = 0,      // Slot is free
    LOADING = 1,    // Prefetch in progress
    READY = 2,      // Data ready for use
    IN_USE = 3      // Currently being accessed
};

/**
 * @brief Single buffer slot for double-buffering
 */
struct BufferSlot {
    void* data;
    size_t size;
    uint32_t tile_index;
    std::atomic<BufferState> state;

    BufferSlot() : data(nullptr), size(0), tile_index(UINT32_MAX),
                   state(BufferState::EMPTY) {}
};

/**
 * @brief Double-buffered cache for BSK tiles
 *
 * Implements ping-pong buffering to overlap prefetch with compute.
 */
class DoubleBufferedCache {
public:
    /**
     * @brief Construct cache with specified slot count
     * @param tile_size Size of each tile in bytes
     * @param num_slots Number of buffer slots (default 4)
     */
    explicit DoubleBufferedCache(size_t tile_size, uint32_t num_slots = 4);

    /**
     * @brief Destructor - frees all allocated memory
     */
    ~DoubleBufferedCache();

    // Non-copyable
    DoubleBufferedCache(const DoubleBufferedCache&) = delete;
    DoubleBufferedCache& operator=(const DoubleBufferedCache&) = delete;

    /**
     * @brief Get a free slot for loading
     * @return Pointer to free slot, or nullptr if none available
     */
    BufferSlot* GetSlotForLoading();

    /**
     * @brief Mark a slot as ready after loading completes
     */
    void MarkSlotReady(BufferSlot* slot, uint32_t tile_index);

    /**
     * @brief Acquire a tile for compute access
     * @param tile_index Tile to acquire
     * @return Pointer to tile data, or nullptr if not cached
     *
     * If tile is still loading, blocks until ready.
     */
    void* AcquireTile(uint32_t tile_index);

    /**
     * @brief Try to acquire tile without blocking
     * @return Pointer to tile data, or nullptr if not available
     */
    void* TryAcquireTile(uint32_t tile_index);

    /**
     * @brief Release a tile after compute completes
     */
    void ReleaseTile(void* data);

    /**
     * @brief Check if tile is cached
     */
    bool IsInCache(uint32_t tile_index) const;

    /**
     * @brief Evict a tile from cache
     */
    void Evict(uint32_t tile_index);

    /**
     * @brief Clear all cached tiles
     */
    void Clear();

    /**
     * @brief Get hit statistics
     */
    uint64_t HitCount() const { return hit_count_.load(); }
    uint64_t MissCount() const { return miss_count_.load(); }
    double HitRate() const;

private:
    size_t tile_size_;
    std::vector<BufferSlot> slots_;
    std::atomic<uint64_t> hit_count_{0};
    std::atomic<uint64_t> miss_count_{0};

    // Memory allocation (platform-specific)
    void* AllocatePinnedMemory(size_t size);
    void FreePinnedMemory(void* ptr, size_t size);
};

// ============================================================================
// Prefetch Statistics
// ============================================================================

/**
 * @brief Statistics for prefetch performance tracking
 */
struct PrefetchStats {
    std::atomic<uint64_t> tiles_prefetched{0};
    std::atomic<uint64_t> cache_hits{0};
    std::atomic<uint64_t> cache_misses{0};
    std::atomic<uint64_t> prefetch_latency_ns{0};
    std::atomic<uint64_t> compute_time_ns{0};
    std::atomic<uint64_t> stall_time_ns{0};

    double HitRate() const {
        uint64_t total = cache_hits + cache_misses;
        if (total == 0) return 1.0;
        return static_cast<double>(cache_hits) / total;
    }

    double PrefetchLag() const {
        if (compute_time_ns == 0) return 0.0;
        return static_cast<double>(prefetch_latency_ns) / compute_time_ns;
    }

    void Reset() {
        tiles_prefetched = 0;
        cache_hits = 0;
        cache_misses = 0;
        prefetch_latency_ns = 0;
        compute_time_ns = 0;
        stall_time_ns = 0;
    }
};

// ============================================================================
// Adaptive Tile Sizer
// ============================================================================

/**
 * @brief Dynamically adjusts tile size based on hit rate feedback
 */
class AdaptiveTileSizer {
public:
    /**
     * @brief Construct with target hit rate
     * @param target_hit_rate Desired cache hit rate (0.0-1.0)
     */
    explicit AdaptiveTileSizer(double target_hit_rate = 0.95);

    /**
     * @brief Record a cache access
     * @param hit True if cache hit, false if miss
     * @param latency_ns Latency of the access in nanoseconds
     */
    void RecordAccess(bool hit, uint64_t latency_ns);

    /**
     * @brief Recommend new tile size based on observed performance
     * @param current_layout Current BSK layout
     * @param device Device memory profile
     * @return Recommended tile_n value
     */
    uint32_t RecommendTileSize(
        const BSKLayout& current_layout,
        const DeviceMemoryProfile& device
    );

    /**
     * @brief Get current statistics
     */
    const PrefetchStats& Stats() const { return stats_; }

    /**
     * @brief Reset statistics
     */
    void Reset();

private:
    double target_hit_rate_;
    PrefetchStats stats_;
    mutable std::mutex mutex_;
};

// ============================================================================
// BSK Prefetcher
// ============================================================================

/**
 * @brief Configuration for BSK prefetcher
 */
struct PrefetcherConfig {
    uint32_t prefetch_ahead;     // How many tiles to prefetch ahead of compute
    uint32_t max_in_flight;      // Maximum concurrent prefetch operations
    double target_hit_rate;      // Target cache hit rate
    bool adaptive_sizing;        // Enable adaptive tile sizing
    bool async_prefetch;         // Use async prefetch thread
    size_t cache_slots;          // Number of cache slots (0 = auto)
};

/**
 * @brief Main BSK prefetcher class
 *
 * Coordinates prediction, prefetch scheduling, and cache management
 * to hide memory latency during blind rotation.
 */
class BSKPrefetcher {
public:
    /**
     * @brief Construct prefetcher with layout and configuration
     */
    BSKPrefetcher(const BSKLayout& layout, const PrefetcherConfig& config);

    /**
     * @brief Destructor - stops prefetch thread
     */
    ~BSKPrefetcher();

    // Non-copyable, non-movable
    BSKPrefetcher(const BSKPrefetcher&) = delete;
    BSKPrefetcher& operator=(const BSKPrefetcher&) = delete;
    BSKPrefetcher(BSKPrefetcher&&) = delete;
    BSKPrefetcher& operator=(BSKPrefetcher&&) = delete;

    /**
     * @brief Start prefetch for a blind rotation operation
     *
     * Analyzes the LWE a-vector and begins prefetching tiles.
     *
     * @param a_vector Pointer to LWE a-vector coefficients
     * @param n Length of a-vector
     */
    void StartPrefetch(const uint64_t* a_vector, uint32_t n);

    /**
     * @brief Acquire a BSK tile for computation
     *
     * Returns cached tile if available, otherwise loads synchronously.
     *
     * @param tile_index Index of tile to acquire
     * @return Pointer to tile data
     */
    void* AcquireTile(uint32_t tile_index);

    /**
     * @brief Release a tile after computation
     */
    void ReleaseTile(void* tile_data);

    /**
     * @brief Notify that an LWE index has been processed
     *
     * Used to adjust prefetch window based on progress.
     */
    void OnLWEIndexProcessed(uint32_t lwe_idx, uint32_t total_n);

    /**
     * @brief Wait for all prefetch operations to complete
     */
    void WaitForCompletion();

    /**
     * @brief Stop prefetch thread
     */
    void Shutdown();

    /**
     * @brief Get current statistics
     */
    const PrefetchStats& Stats() const;

    /**
     * @brief Reset statistics
     */
    void ResetStats();

    /**
     * @brief Get current layout
     */
    const BSKLayout& Layout() const { return layout_; }

    /**
     * @brief Update layout (e.g., after adaptive resizing)
     */
    void UpdateLayout(const BSKLayout& layout);

protected:
    /**
     * @brief Prefetch thread main loop
     */
    void PrefetchLoop();

    /**
     * @brief Issue an asynchronous prefetch
     */
    virtual void IssuePrefetchAsync(
        const PrefetchRequest& request,
        std::function<void()> completion
    );

    /**
     * @brief Synchronous tile load (fallback on cache miss)
     */
    virtual void* SynchronousLoad(uint32_t tile_index);

private:
    BSKLayout layout_;
    PrefetcherConfig config_;

    PrefetchQueue queue_;
    std::unique_ptr<DoubleBufferedCache> cache_;
    AdaptiveTileSizer sizer_;
    PrefetchStats stats_;

    std::atomic<uint32_t> in_flight_{0};
    std::atomic<uint32_t> current_lwe_index_{0};
    std::atomic<bool> shutdown_{false};
    std::atomic<bool> operation_active_{false};

    std::thread prefetch_thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

// ============================================================================
// Prefetching Blind Rotation Integration
// ============================================================================

/**
 * @brief CGGI accumulator with integrated prefetching
 *
 * Wraps the standard CGGI blind rotation to add prefetch support.
 */
class PrefetchingAccumulator {
public:
    /**
     * @brief Construct with prefetcher
     */
    explicit PrefetchingAccumulator(std::shared_ptr<BSKPrefetcher> prefetcher);

    /**
     * @brief Set the BSK memory pointers
     */
    void SetBSK(void* bsk_ptr, const BSKLayout& layout);

    /**
     * @brief Get prefetcher
     */
    BSKPrefetcher* Prefetcher() { return prefetcher_.get(); }

    /**
     * @brief Extract RGSW key from tile for given LWE index
     *
     * @param tile_data Pointer to tile data
     * @param lwe_idx LWE index within the BSK
     * @param sign_idx Sign index (0 or 1)
     * @return Pointer to RGSW ciphertext within tile
     */
    const void* ExtractFromTile(
        const void* tile_data,
        uint32_t lwe_idx,
        uint32_t sign_idx
    ) const;

protected:
    std::shared_ptr<BSKPrefetcher> prefetcher_;
    BSKLayout layout_;
};

} // namespace backend
} // namespace lux::fhe

#endif // BACKEND_BSK_PREFETCH_H
