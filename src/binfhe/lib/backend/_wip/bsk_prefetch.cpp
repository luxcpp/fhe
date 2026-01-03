// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// Speculative Bootstrapping Key Prefetcher - Implementation

#include "backend/bsk_prefetch.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <unordered_set>

#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/mman.h>
#endif

#ifdef __linux__
#include <sys/mman.h>
#include <numa.h>
#endif

namespace lux::fhe {
namespace backend {

// ============================================================================
// Device Memory Profile
// ============================================================================

uint32_t DeviceMemoryProfile::OptimalTileN(const BSKLayout& layout) const {
    // Target: tile fits in L2 cache with room for computation working set
    size_t available_cache = static_cast<size_t>(l2_cache_size * 0.6);

    // Size per LWE index
    size_t per_index_size = layout.PerIndexSize();

    // Compute maximum indices that fit
    uint32_t max_indices = static_cast<uint32_t>(available_cache / per_index_size);
    max_indices = std::max(max_indices, 1u);
    max_indices = std::min(max_indices, layout.n);

    // Round down to power of 2 for alignment
    uint32_t tile_n = 1;
    while (tile_n * 2 <= max_indices) {
        tile_n *= 2;
    }

    return tile_n;
}

DeviceMemoryProfile DeviceMemoryProfile::M1Max() {
    DeviceMemoryProfile profile;
    profile.l1_cache_size = 192 * 1024;         // 192 KB per performance core
    profile.l2_cache_size = 48 * 1024 * 1024;   // 48 MB shared
    profile.shared_memory_size = 32 * 1024;     // 32 KB threadgroup memory
    profile.register_file_size = 256 * 1024;    // Estimated

    profile.hbm_bandwidth = 400.0;              // 400 GB/s unified memory
    profile.l2_bandwidth = 2000.0;              // ~2 TB/s L2 bandwidth
    profile.compute_throughput = 10600.0;       // 10.6 TFLOPS FP32

    profile.memory_latency_us = 0.1;            // ~100ns

    return profile;
}

DeviceMemoryProfile DeviceMemoryProfile::A100() {
    DeviceMemoryProfile profile;
    profile.l1_cache_size = 192 * 1024;         // 192 KB per SM
    profile.l2_cache_size = 40 * 1024 * 1024;   // 40 MB L2
    profile.shared_memory_size = 164 * 1024;    // 164 KB shared per SM
    profile.register_file_size = 256 * 1024;    // 256 KB per SM

    profile.hbm_bandwidth = 2000.0;             // 2 TB/s HBM2e
    profile.l2_bandwidth = 5000.0;              // ~5 TB/s L2 bandwidth
    profile.compute_throughput = 19500.0;       // 19.5 TFLOPS FP32

    profile.memory_latency_us = 0.4;            // ~400ns HBM latency

    return profile;
}

DeviceMemoryProfile DeviceMemoryProfile::CPU() {
    DeviceMemoryProfile profile;
    profile.l1_cache_size = 64 * 1024;          // 64 KB per core
    profile.l2_cache_size = 256 * 1024;         // 256 KB per core
    profile.shared_memory_size = 0;             // No shared memory concept
    profile.register_file_size = 0;

    // Assume high-end server with DDR5
    profile.hbm_bandwidth = 200.0;              // DDR5-4800 quad-channel
    profile.l2_bandwidth = 500.0;               // L2 bandwidth
    profile.compute_throughput = 2000.0;        // ~2 TFLOPS with AVX-512

    profile.memory_latency_us = 0.08;           // ~80ns DDR latency

    return profile;
}

// ============================================================================
// Access Pattern Prediction
// ============================================================================

std::vector<TileAccessPrediction> PredictAccessSequence(
    const uint64_t* a_vector,
    uint32_t n,
    const BSKLayout& layout
) {
    std::vector<TileAccessPrediction> predictions;
    predictions.reserve(layout.num_tiles);

    // Track first access to each tile
    std::vector<int32_t> first_access(layout.num_tiles, -1);

    for (uint32_t i = 0; i < n; ++i) {
        // Compute which tile this LWE index maps to
        uint32_t tile_idx = layout.TileIndex(i);

        if (first_access[tile_idx] < 0) {
            first_access[tile_idx] = static_cast<int32_t>(i);

            TileAccessPrediction pred;
            pred.tile_index = tile_idx;
            pred.lwe_index = i;
            pred.priority = i;  // Access order is priority

            predictions.push_back(pred);
        }
    }

    // Sort by access order (priority)
    std::sort(predictions.begin(), predictions.end(),
        [](const TileAccessPrediction& a, const TileAccessPrediction& b) {
            return a.priority < b.priority;
        });

    return predictions;
}

// ============================================================================
// Prefetch Queue
// ============================================================================

void PrefetchQueue::EnqueuePredictions(
    const std::vector<TileAccessPrediction>& predictions,
    const BSKLayout& layout
) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (const auto& pred : predictions) {
        // Skip if already queued
        if (queued_tiles_.count(pred.tile_index)) {
            continue;
        }

        PrefetchRequest req;
        req.tile_index = pred.tile_index;
        req.priority = pred.priority;
        req.source_ptr = layout.TilePtr(pred.tile_index);
        req.dest_ptr = nullptr;  // Will be assigned when dequeued
        req.size = layout.tile_size;

        queue_.push(req);
        queued_tiles_.insert(pred.tile_index);
    }
}

std::optional<PrefetchRequest> PrefetchQueue::PopHighestPriority() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (queue_.empty()) {
        return std::nullopt;
    }

    PrefetchRequest req = queue_.top();
    queue_.pop();
    // Keep in queued_tiles_ until completed

    return req;
}

bool PrefetchQueue::IsQueued(uint32_t tile_index) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queued_tiles_.count(tile_index) > 0;
}

bool PrefetchQueue::Empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

void PrefetchQueue::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!queue_.empty()) {
        queue_.pop();
    }
    queued_tiles_.clear();
}

size_t PrefetchQueue::Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

// ============================================================================
// Double-Buffered Cache
// ============================================================================

DoubleBufferedCache::DoubleBufferedCache(size_t tile_size, uint32_t num_slots)
    : tile_size_(tile_size), slots_(num_slots) {

    for (auto& slot : slots_) {
        slot.data = AllocatePinnedMemory(tile_size);
        slot.size = tile_size;
        slot.tile_index = UINT32_MAX;
        slot.state = BufferState::EMPTY;
    }
}

DoubleBufferedCache::~DoubleBufferedCache() {
    for (auto& slot : slots_) {
        if (slot.data) {
            FreePinnedMemory(slot.data, slot.size);
            slot.data = nullptr;
        }
    }
}

BufferSlot* DoubleBufferedCache::GetSlotForLoading() {
    for (auto& slot : slots_) {
        BufferState expected = BufferState::EMPTY;
        if (slot.state.compare_exchange_strong(expected, BufferState::LOADING)) {
            return &slot;
        }
    }
    return nullptr;  // No free slots
}

void DoubleBufferedCache::MarkSlotReady(BufferSlot* slot, uint32_t tile_index) {
    if (slot) {
        slot->tile_index = tile_index;
        slot->state = BufferState::READY;
    }
}

void* DoubleBufferedCache::AcquireTile(uint32_t tile_index) {
    // First check if already in a slot
    for (auto& slot : slots_) {
        if (slot.tile_index == tile_index) {
            // Found it - wait if still loading
            BufferState current = slot.state.load();

            if (current == BufferState::LOADING) {
                // Spin-wait for loading to complete
                while ((current = slot.state.load()) == BufferState::LOADING) {
                    std::this_thread::yield();
                }
            }

            // Try to acquire
            BufferState expected = BufferState::READY;
            if (slot.state.compare_exchange_strong(expected, BufferState::IN_USE)) {
                hit_count_++;
                return slot.data;
            }

            // Already in use by someone else - wait
            while ((current = slot.state.load()) == BufferState::IN_USE) {
                std::this_thread::yield();
            }

            // Try again
            expected = BufferState::READY;
            if (slot.state.compare_exchange_strong(expected, BufferState::IN_USE)) {
                hit_count_++;
                return slot.data;
            }
        }
    }

    // Cache miss
    miss_count_++;
    return nullptr;
}

void* DoubleBufferedCache::TryAcquireTile(uint32_t tile_index) {
    for (auto& slot : slots_) {
        if (slot.tile_index == tile_index) {
            BufferState expected = BufferState::READY;
            if (slot.state.compare_exchange_strong(expected, BufferState::IN_USE)) {
                hit_count_++;
                return slot.data;
            }
            return nullptr;  // Not ready yet
        }
    }
    return nullptr;  // Not in cache
}

void DoubleBufferedCache::ReleaseTile(void* data) {
    for (auto& slot : slots_) {
        if (slot.data == data) {
            slot.state = BufferState::EMPTY;
            slot.tile_index = UINT32_MAX;
            return;
        }
    }
}

bool DoubleBufferedCache::IsInCache(uint32_t tile_index) const {
    for (const auto& slot : slots_) {
        if (slot.tile_index == tile_index) {
            BufferState state = slot.state.load();
            return state == BufferState::READY || state == BufferState::IN_USE;
        }
    }
    return false;
}

void DoubleBufferedCache::Evict(uint32_t tile_index) {
    for (auto& slot : slots_) {
        if (slot.tile_index == tile_index) {
            // Wait for any in-use to complete
            BufferState expected = BufferState::IN_USE;
            while (slot.state.load() == BufferState::IN_USE) {
                std::this_thread::yield();
            }

            expected = BufferState::READY;
            if (slot.state.compare_exchange_strong(expected, BufferState::EMPTY)) {
                slot.tile_index = UINT32_MAX;
            }
            return;
        }
    }
}

void DoubleBufferedCache::Clear() {
    for (auto& slot : slots_) {
        // Wait for in-use slots
        while (slot.state.load() == BufferState::IN_USE) {
            std::this_thread::yield();
        }

        slot.state = BufferState::EMPTY;
        slot.tile_index = UINT32_MAX;
    }
}

double DoubleBufferedCache::HitRate() const {
    uint64_t total = hit_count_ + miss_count_;
    if (total == 0) return 1.0;
    return static_cast<double>(hit_count_) / total;
}

void* DoubleBufferedCache::AllocatePinnedMemory(size_t size) {
#ifdef __APPLE__
    // macOS: use mmap with MAP_ANONYMOUS for page-aligned memory
    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        return nullptr;
    }
    // Wire pages to prevent paging
    mlock(ptr, size);
    return ptr;
#elif defined(__linux__)
    // Linux: use mmap with huge pages if available
    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (ptr == MAP_FAILED) {
        return nullptr;
    }
    mlock(ptr, size);
    return ptr;
#else
    // Fallback: aligned allocation
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 4096, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

void DoubleBufferedCache::FreePinnedMemory(void* ptr, size_t size) {
    if (!ptr) return;

#ifdef __APPLE__
    munlock(ptr, size);
    munmap(ptr, size);
#elif defined(__linux__)
    munlock(ptr, size);
    munmap(ptr, size);
#else
    free(ptr);
#endif
}

// ============================================================================
// Adaptive Tile Sizer
// ============================================================================

AdaptiveTileSizer::AdaptiveTileSizer(double target_hit_rate)
    : target_hit_rate_(target_hit_rate) {
}

void AdaptiveTileSizer::RecordAccess(bool hit, uint64_t latency_ns) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (hit) {
        stats_.cache_hits++;
    } else {
        stats_.cache_misses++;
        stats_.prefetch_latency_ns += latency_ns;
    }
}

uint32_t AdaptiveTileSizer::RecommendTileSize(
    const BSKLayout& current_layout,
    const DeviceMemoryProfile& device
) {
    std::lock_guard<std::mutex> lock(mutex_);

    double hit_rate = stats_.HitRate();
    double prefetch_lag = stats_.PrefetchLag();

    // Target bounds
    const double low_threshold = 0.90;
    const double high_threshold = 0.99;

    if (hit_rate < low_threshold) {
        // Too many misses
        if (prefetch_lag > 0.5) {
            // Prefetch falling behind - increase tile size to reduce request count
            uint32_t new_tile_n = current_layout.tile_n * 2;
            return std::min(new_tile_n, current_layout.n);
        } else {
            // Cache thrashing - decrease tile size to reduce pressure
            uint32_t new_tile_n = current_layout.tile_n / 2;
            return std::max(new_tile_n, 1u);
        }
    } else if (hit_rate > high_threshold) {
        // Potentially over-caching - increase tile size to reduce overhead
        uint32_t new_tile_n = current_layout.tile_n * 2;
        return std::min(new_tile_n, current_layout.n);
    }

    // Hit rate is good, maintain current size
    return current_layout.tile_n;
}

void AdaptiveTileSizer::Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.Reset();
}

// ============================================================================
// BSK Prefetcher
// ============================================================================

BSKPrefetcher::BSKPrefetcher(const BSKLayout& layout, const PrefetcherConfig& config)
    : layout_(layout)
    , config_(config)
    , sizer_(config.target_hit_rate) {

    // Compute cache slots if not specified
    size_t cache_slots = config.cache_slots;
    if (cache_slots == 0) {
        // Default: enough slots to prefetch ahead + some buffer
        cache_slots = config.prefetch_ahead * 2 + 2;
    }

    cache_ = std::make_unique<DoubleBufferedCache>(layout.tile_size, cache_slots);

    // Start prefetch thread if async enabled
    if (config.async_prefetch) {
        prefetch_thread_ = std::thread(&BSKPrefetcher::PrefetchLoop, this);
    }
}

BSKPrefetcher::~BSKPrefetcher() {
    Shutdown();
}

void BSKPrefetcher::StartPrefetch(const uint64_t* a_vector, uint32_t n) {
    // Predict access sequence
    auto predictions = PredictAccessSequence(a_vector, n, layout_);

    // Clear any pending from previous operation
    queue_.Clear();

    // Enqueue predictions
    queue_.EnqueuePredictions(predictions, layout_);

    // Reset progress tracking
    current_lwe_index_ = 0;
    operation_active_ = true;

    // Wake prefetch thread
    cv_.notify_all();
}

void* BSKPrefetcher::AcquireTile(uint32_t tile_index) {
    auto start = std::chrono::high_resolution_clock::now();

    // Try cache first
    void* data = cache_->AcquireTile(tile_index);

    if (data) {
        sizer_.RecordAccess(true, 0);
        stats_.cache_hits++;
        return data;
    }

    // Cache miss - synchronous load
    data = SynchronousLoad(tile_index);

    auto end = std::chrono::high_resolution_clock::now();
    uint64_t latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end - start).count();

    sizer_.RecordAccess(false, latency_ns);
    stats_.cache_misses++;
    stats_.stall_time_ns += latency_ns;

    return data;
}

void BSKPrefetcher::ReleaseTile(void* tile_data) {
    cache_->ReleaseTile(tile_data);
}

void BSKPrefetcher::OnLWEIndexProcessed(uint32_t lwe_idx, uint32_t total_n) {
    current_lwe_index_ = lwe_idx;

    // Adjust prefetch window near end
    uint32_t remaining = total_n - lwe_idx;
    if (remaining < config_.prefetch_ahead * 2 && config_.adaptive_sizing) {
        // Could dynamically adjust here if needed
    }
}

void BSKPrefetcher::WaitForCompletion() {
    operation_active_ = false;

    // Wait for in-flight prefetches
    while (in_flight_ > 0) {
        std::this_thread::yield();
    }
}

void BSKPrefetcher::Shutdown() {
    if (shutdown_.exchange(true)) {
        return;  // Already shutdown
    }

    operation_active_ = false;
    cv_.notify_all();

    if (prefetch_thread_.joinable()) {
        prefetch_thread_.join();
    }
}

const PrefetchStats& BSKPrefetcher::Stats() const {
    return stats_;
}

void BSKPrefetcher::ResetStats() {
    stats_.Reset();
    sizer_.Reset();
}

void BSKPrefetcher::UpdateLayout(const BSKLayout& layout) {
    layout_ = layout;
    cache_->Clear();
}

void BSKPrefetcher::PrefetchLoop() {
    while (!shutdown_) {
        // Wait for operation to start
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] {
                return shutdown_ || (!queue_.Empty() && operation_active_);
            });
        }

        if (shutdown_) break;

        // Process prefetch requests
        while (operation_active_ && !queue_.Empty()) {
            // Limit in-flight requests
            while (in_flight_ >= config_.max_in_flight && !shutdown_) {
                std::this_thread::yield();
            }

            if (shutdown_) break;

            auto req = queue_.PopHighestPriority();
            if (!req) {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                continue;
            }

            // Skip if already in cache
            if (cache_->IsInCache(req->tile_index)) {
                continue;
            }

            // Get a slot for loading
            BufferSlot* slot = cache_->GetSlotForLoading();
            if (!slot) {
                // No free slots - wait
                std::this_thread::yield();
                continue;
            }

            // Update request with destination
            req->dest_ptr = slot->data;

            // Issue async prefetch
            in_flight_++;

            IssuePrefetchAsync(*req, [this, slot, tile_idx = req->tile_index]() {
                cache_->MarkSlotReady(slot, tile_idx);
                in_flight_--;
                stats_.tiles_prefetched++;
            });
        }
    }
}

void BSKPrefetcher::IssuePrefetchAsync(
    const PrefetchRequest& request,
    std::function<void()> completion
) {
    // Default implementation: synchronous memcpy in background
    // Subclasses override for platform-specific async (Metal, CUDA)
    std::memcpy(request.dest_ptr, request.source_ptr, request.size);
    completion();
}

void* BSKPrefetcher::SynchronousLoad(uint32_t tile_index) {
    // Get a slot
    BufferSlot* slot = nullptr;
    while (!(slot = cache_->GetSlotForLoading())) {
        // Wait for a slot to become available
        std::this_thread::yield();
    }

    // Load data
    void* src = layout_.TilePtr(tile_index);
    std::memcpy(slot->data, src, layout_.tile_size);

    // Mark ready
    cache_->MarkSlotReady(slot, tile_index);

    // Acquire for use
    slot->state = BufferState::IN_USE;

    return slot->data;
}

// ============================================================================
// Prefetching Accumulator
// ============================================================================

PrefetchingAccumulator::PrefetchingAccumulator(std::shared_ptr<BSKPrefetcher> prefetcher)
    : prefetcher_(std::move(prefetcher)) {
    if (prefetcher_) {
        layout_ = prefetcher_->Layout();
    }
}

void PrefetchingAccumulator::SetBSK(void* bsk_ptr, const BSKLayout& layout) {
    layout_ = layout;
    layout_.base_ptr = bsk_ptr;

    if (prefetcher_) {
        prefetcher_->UpdateLayout(layout_);
    }
}

const void* PrefetchingAccumulator::ExtractFromTile(
    const void* tile_data,
    uint32_t lwe_idx,
    uint32_t sign_idx
) const {
    // Compute offset within tile
    uint32_t tile_offset = layout_.TileOffset(lwe_idx);

    // Each LWE index has 2 RGSW ciphertexts (sign 0 and 1)
    size_t element_size = layout_.ElementSize();
    size_t per_index_size = layout_.PerIndexSize();

    // Offset: tile_offset indices * per_index_size + sign_idx * element_size
    size_t byte_offset = tile_offset * per_index_size + sign_idx * element_size;

    return static_cast<const uint8_t*>(tile_data) + byte_offset;
}

} // namespace backend
} // namespace lux::fhe
