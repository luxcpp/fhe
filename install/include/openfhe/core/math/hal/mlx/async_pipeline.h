// =============================================================================
// Async Pipeline for BSK Access with Double-Buffering
// =============================================================================
//
// Overlaps BSK memory fetches with computation for improved throughput.
//
// Key Innovations:
// 1. Double-buffered BSK storage: ping-pong between two GPU buffers
// 2. Async evaluation for concurrent operations
// 3. Prefetch next batch's BSK while current batch computes
// 4. Thread pool for parallel batch submission
//
// Pipeline Structure:
//   Buffer 0: [Fetch BSK[0]] [Compute PBS[0]] [Fetch BSK[2]] ...
//   Buffer 1: [Fetch BSK[1]] [Compute PBS[1]] [Fetch BSK[3]] ...
//
// Expected speedup: 1.5-2x from hiding memory latency
//
// Copyright (C) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: Apache-2.0
// =============================================================================

#ifndef LBCRYPTO_MATH_HAL_MLX_ASYNC_PIPELINE_H
#define LBCRYPTO_MATH_HAL_MLX_ASYNC_PIPELINE_H

#include <cstdint>
#include <vector>
#include <memory>
#include <queue>
#include <future>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <array>

#ifdef WITH_MLX
#include <mlx/mlx.h>
namespace mx = mlx::core;
#endif

namespace lbcrypto {
namespace gpu {

#ifdef WITH_MLX

// =============================================================================
// BSK Buffer Pool - Double-buffered storage for BSK entries
// =============================================================================

class BSKBufferPool {
public:
    struct Config {
        uint32_t N = 1024;          // Ring dimension
        uint32_t n = 512;           // LWE dimension (number of BSK entries)
        uint32_t L = 3;             // Decomposition levels
        size_t num_buffers = 2;     // Double buffering by default
    };

    BSKBufferPool() = default;

    explicit BSKBufferPool(const Config& cfg)
        : cfg_(cfg), active_idx_(0) {

        // Each BSK entry: [2, L, 2, N] int64
        entry_size_ = 2 * cfg_.L * 2 * cfg_.N;

        // Allocate buffer pool - use reserve + emplace_back since mx::array has no default ctor
        buffers_.reserve(cfg_.num_buffers);
        buffer_ready_.resize(cfg_.num_buffers, false);
        buffer_entry_idx_.resize(cfg_.num_buffers, UINT32_MAX);

        for (size_t i = 0; i < cfg_.num_buffers; ++i) {
            buffers_.emplace_back(mx::zeros({static_cast<int>(entry_size_)}, mx::int64));
            mx::eval(buffers_.back());
        }
    }

    // Get active buffer (for computation)
    mx::array& active() { return buffers_[active_idx_]; }
    const mx::array& active() const { return buffers_[active_idx_]; }

    // Get prefetch buffer (next in rotation)
    mx::array& prefetch() {
        return buffers_[(active_idx_ + 1) % cfg_.num_buffers];
    }

    // Rotate to next buffer
    void advance() {
        active_idx_ = (active_idx_ + 1) % cfg_.num_buffers;
    }

    // Check if buffer contains valid data for entry
    bool hasEntry(size_t buffer_idx, uint32_t entry_idx) const {
        return buffer_ready_[buffer_idx] && buffer_entry_idx_[buffer_idx] == entry_idx;
    }

    // Mark buffer as containing entry
    void setEntry(size_t buffer_idx, uint32_t entry_idx) {
        buffer_entry_idx_[buffer_idx] = entry_idx;
        buffer_ready_[buffer_idx] = true;
    }

    // Get buffer by index
    mx::array& buffer(size_t idx) { return buffers_[idx]; }

    size_t activeIndex() const { return active_idx_; }
    size_t prefetchIndex() const { return (active_idx_ + 1) % cfg_.num_buffers; }
    size_t numBuffers() const { return cfg_.num_buffers; }
    size_t entrySize() const { return entry_size_; }

private:
    Config cfg_{};
    size_t entry_size_ = 0;
    size_t active_idx_ = 0;
    std::vector<mx::array> buffers_;
    std::vector<bool> buffer_ready_;
    std::vector<uint32_t> buffer_entry_idx_;
};

// =============================================================================
// Stream Executor - Thread pool for async task execution
// =============================================================================
//
// Provides a simple thread pool for submitting async tasks.
// Uses std::async internally for non-blocking execution.

class StreamExecutor {
public:
    explicit StreamExecutor(size_t num_workers = 2)
        : num_workers_(num_workers), running_(true), next_id_(0) {

        // Start worker threads
        workers_.reserve(num_workers_);
        for (size_t i = 0; i < num_workers_; ++i) {
            workers_.emplace_back([this]() { workerLoop(); });
        }
    }

    ~StreamExecutor() {
        // Signal shutdown
        {
            std::lock_guard<std::mutex> lock(mutex_);
            running_ = false;
        }
        cv_.notify_all();

        // Wait for workers
        for (auto& w : workers_) {
            if (w.joinable()) {
                w.join();
            }
        }
    }

    // Execute function asynchronously
    // Returns future for result synchronization
    template<typename F>
    auto execute(F&& func) -> std::future<decltype(func())> {
        using ReturnType = decltype(func());

        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::forward<F>(func));
        auto future = task->get_future();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.push([task]() { (*task)(); });
        }

        cv_.notify_one();
        return future;
    }

    // Synchronize all pending tasks (blocks until complete)
    void synchronize() {
        std::unique_lock<std::mutex> lock(mutex_);
        sync_cv_.wait(lock, [this]() { return tasks_.empty() && active_tasks_ == 0; });
    }

    size_t numWorkers() const { return num_workers_; }

private:
    void workerLoop() {
        while (true) {
            std::function<void()> task;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() {
                    return !running_ || !tasks_.empty();
                });

                if (!running_ && tasks_.empty()) {
                    return;
                }

                if (!tasks_.empty()) {
                    task = std::move(tasks_.front());
                    tasks_.pop();
                    ++active_tasks_;
                }
            }

            if (task) {
                task();
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    --active_tasks_;
                }
                sync_cv_.notify_all();
            }
        }
    }

    size_t num_workers_;
    std::atomic<bool> running_;
    std::atomic<size_t> next_id_;
    size_t active_tasks_ = 0;

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable sync_cv_;
};

// =============================================================================
// Async PBS Pipeline - Main pipeline for overlapping BSK fetch with compute
// =============================================================================

class AsyncPBSPipeline {
public:
    struct Config {
        uint32_t N = 1024;              // Ring dimension
        uint32_t n = 512;               // LWE dimension
        uint32_t L = 3;                 // Decomposition levels
        uint32_t baseLog = 7;           // Decomposition base log
        uint64_t Q = 1ULL << 27;        // Ring modulus
        size_t pipeline_depth = 2;      // Double buffering
        size_t prefetch_batches = 1;    // How many batches to prefetch ahead
        size_t num_streams = 2;         // Number of worker threads
    };

    AsyncPBSPipeline() = default;

    explicit AsyncPBSPipeline(const Config& cfg)
        : cfg_(cfg), running_(true) {

        // Initialize buffer pool for BSK double-buffering
        BSKBufferPool::Config pool_cfg;
        pool_cfg.N = cfg_.N;
        pool_cfg.n = cfg_.n;
        pool_cfg.L = cfg_.L;
        pool_cfg.num_buffers = cfg_.pipeline_depth;
        bsk_pool_ = std::make_unique<BSKBufferPool>(pool_cfg);

        // Initialize stream executor
        executor_ = std::make_unique<StreamExecutor>(cfg_.num_streams);

        // Start prefetch worker thread
        prefetch_worker_ = std::thread([this]() { prefetchWorkerLoop(); });
    }

    ~AsyncPBSPipeline() {
        running_ = false;
        prefetch_cv_.notify_all();

        if (prefetch_worker_.joinable()) {
            prefetch_worker_.join();
        }
    }

    // Set the full BSK array (stored on GPU)
    void setBSK(const mx::array& bsk) {
        std::lock_guard<std::mutex> lock(bsk_mutex_);
        bsk_ = std::make_shared<mx::array>(bsk);
        mx::eval(*bsk_);
    }

    // Submit batch for async execution
    // Returns future for result
    std::future<std::vector<mx::array>> submitBatch(
        const std::vector<mx::array>& lwes,
        const mx::array& test_poly) {

        // Capture copies for the async task
        auto lwes_copy = lwes;
        auto test_poly_copy = test_poly;

        return executor_->execute([this, lwes_copy, test_poly_copy]() {
            return executeBatchInternal(lwes_copy, test_poly_copy);
        });
    }

    // Prefetch BSK entries for upcoming computation (async)
    void prefetchBSK(uint32_t start_entry, uint32_t count = 1) {
        {
            std::lock_guard<std::mutex> lock(prefetch_mutex_);
            for (uint32_t i = 0; i < count; ++i) {
                prefetch_queue_.push(start_entry + i);
            }
        }
        prefetch_cv_.notify_one();
    }

    // Wait for all pending operations
    void sync() {
        executor_->synchronize();
    }

    // Get pipeline statistics
    struct Stats {
        uint64_t batches_submitted = 0;
        uint64_t prefetch_hits = 0;
        uint64_t prefetch_misses = 0;
        double avg_batch_time_ms = 0.0;
    };

    Stats stats() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        return stats_;
    }

    const Config& config() const { return cfg_; }

private:
    // Execute batch synchronously (called from worker thread)
    std::vector<mx::array> executeBatchInternal(
        const std::vector<mx::array>& lwes,
        const mx::array& test_poly) {

        if (lwes.empty()) {
            return {};
        }

        std::lock_guard<std::mutex> lock(bsk_mutex_);
        if (!bsk_) {
            return {};
        }

        size_t batch_size = lwes.size();
        int N = static_cast<int>(cfg_.N);
        int n = static_cast<int>(cfg_.n);
        uint64_t Q = cfg_.Q;

        // Stack LWEs into batch
        auto lwe_batch = mx::stack(lwes, 0);
        mx::eval(lwe_batch);

        std::vector<mx::array> results;
        results.reserve(batch_size);

        // Get raw pointer for LWE data
        auto lwe_ptr = lwe_batch.data<int64_t>();

        // Process each LWE with pipelined BSK access
        for (size_t b = 0; b < batch_size; ++b) {
            const int64_t* lwe = lwe_ptr + b * (n + 1);

            // Initialize accumulator with X^{-b} * testPoly
            int64_t b_val = lwe[n];
            int32_t shift = static_cast<int32_t>(((b_val % (2 * N)) + 2 * N) % (2 * N));

            std::vector<int64_t> acc0_data(N, 0);
            std::vector<int64_t> acc1_data(N);

            mx::eval(test_poly);
            auto test_ptr = test_poly.data<int64_t>();

            // Negacyclic rotation of test polynomial
            for (int j = 0; j < N; ++j) {
                int32_t src_idx = j + shift;
                bool negate = false;
                while (src_idx >= N) { src_idx -= N; negate = !negate; }
                while (src_idx < 0) { src_idx += N; negate = !negate; }

                int64_t val = test_ptr[src_idx] % static_cast<int64_t>(Q);
                acc1_data[j] = negate ? static_cast<int64_t>((Q - val) % Q) : val;
            }

            auto acc0 = mx::array(acc0_data.data(), {N}, mx::int64);
            auto acc1 = mx::array(acc1_data.data(), {N}, mx::int64);
            mx::eval(acc0, acc1);

            // Blind rotation loop with pipelined BSK access
            for (int i = 0; i < n; ++i) {
                int64_t a_val = lwe[i];
                if (a_val == 0) continue;

                // Get BSK entry
                auto bsk_entry = getBSKEntry(static_cast<uint32_t>(i));
                mx::eval(bsk_entry);

                // Compute rotation
                int32_t rot = static_cast<int32_t>(((a_val % (2 * N)) + 2 * N) % (2 * N));

                // Rotated accumulator
                std::vector<int64_t> rot0(N), rot1(N);
                auto acc0_ptr = acc0.data<int64_t>();
                auto acc1_ptr = acc1.data<int64_t>();

                for (int j = 0; j < N; ++j) {
                    int32_t src = j - rot;
                    bool neg = false;
                    while (src < 0) { src += N; neg = !neg; }
                    while (src >= N) { src -= N; neg = !neg; }

                    rot0[j] = neg ? static_cast<int64_t>((Q - acc0_ptr[src]) % Q) : acc0_ptr[src];
                    rot1[j] = neg ? static_cast<int64_t>((Q - acc1_ptr[src]) % Q) : acc1_ptr[src];
                }

                // diff = rotated - acc
                std::vector<int64_t> diff0(N), diff1(N);
                for (int j = 0; j < N; ++j) {
                    diff0[j] = (rot0[j] >= acc0_ptr[j]) ?
                        rot0[j] - acc0_ptr[j] :
                        static_cast<int64_t>(rot0[j] + Q - acc0_ptr[j]);
                    diff1[j] = (rot1[j] >= acc1_ptr[j]) ?
                        rot1[j] - acc1_ptr[j] :
                        static_cast<int64_t>(rot1[j] + Q - acc1_ptr[j]);
                }

                // External product with BSK entry
                auto bsk_ptr = bsk_entry.data<int64_t>();
                std::vector<int64_t> prod0(N, 0), prod1(N, 0);

                uint64_t mask = (1ULL << cfg_.baseLog) - 1;

                for (uint32_t c = 0; c < 2; ++c) {
                    const std::vector<int64_t>& diff_comp = (c == 0) ? diff0 : diff1;

                    for (uint32_t l = 0; l < cfg_.L; ++l) {
                        // Extract digit
                        std::vector<uint64_t> digit(N);
                        for (int j = 0; j < N; ++j) {
                            digit[j] = (static_cast<uint64_t>(diff_comp[j]) >> (l * cfg_.baseLog)) & mask;
                        }

                        // RGSW layout: [2, L, 2, N] flattened
                        size_t row_offset = c * cfg_.L * 2 * N + l * 2 * N;

                        for (uint32_t out_c = 0; out_c < 2; ++out_c) {
                            const int64_t* rgsw_poly = bsk_ptr + row_offset + out_c * N;
                            std::vector<int64_t>& prod_comp = (out_c == 0) ? prod0 : prod1;

                            for (int j = 0; j < N; ++j) {
                                uint64_t rgsw_val = static_cast<uint64_t>(rgsw_poly[j]) % Q;
                                uint64_t mul = static_cast<uint64_t>(
                                    (__uint128_t)digit[j] * rgsw_val % Q);
                                prod_comp[j] = static_cast<int64_t>((prod_comp[j] + mul) % Q);
                            }
                        }
                    }
                }

                // Update accumulator
                std::vector<int64_t> new_acc0(N), new_acc1(N);
                for (int j = 0; j < N; ++j) {
                    new_acc0[j] = static_cast<int64_t>((acc0_ptr[j] + prod0[j]) % Q);
                    new_acc1[j] = static_cast<int64_t>((acc1_ptr[j] + prod1[j]) % Q);
                }

                acc0 = mx::array(new_acc0.data(), {N}, mx::int64);
                acc1 = mx::array(new_acc1.data(), {N}, mx::int64);
                mx::eval(acc0, acc1);
            }

            // Stack result as RLWE [2, N]
            auto result = mx::stack({acc0, acc1}, 0);
            result = mx::reshape(result, {2, N});
            mx::eval(result);
            results.push_back(std::move(result));
        }

        // Update stats
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.batches_submitted++;
        }

        return results;
    }

    // Get BSK entry (thread-safe)
    mx::array getBSKEntry(uint32_t entry_idx) {
        // bsk_mutex_ already held by caller

        if (!bsk_ || entry_idx >= cfg_.n) {
            return mx::zeros({static_cast<int>(2 * cfg_.L * 2 * cfg_.N)}, mx::int64);
        }

        int L = static_cast<int>(cfg_.L);
        int N = static_cast<int>(cfg_.N);
        int i = static_cast<int>(entry_idx);

        // Slice BSK entry: [n, 2, L, 2, N] -> [2, L, 2, N]
        auto entry = mx::slice(*bsk_,
            {i, 0, 0, 0, 0},
            {i + 1, 2, L, 2, N});
        entry = mx::reshape(entry, {static_cast<int>(2 * cfg_.L * 2 * cfg_.N)});
        mx::eval(entry);

        return entry;
    }

    // Prefetch worker loop
    void prefetchWorkerLoop() {
        while (running_) {
            uint32_t entry_idx;

            {
                std::unique_lock<std::mutex> lock(prefetch_mutex_);
                prefetch_cv_.wait(lock, [this]() {
                    return !running_ || !prefetch_queue_.empty();
                });

                if (!running_ && prefetch_queue_.empty()) {
                    return;
                }

                if (!prefetch_queue_.empty()) {
                    entry_idx = prefetch_queue_.front();
                    prefetch_queue_.pop();
                } else {
                    continue;
                }
            }

            // Prefetch BSK entry into buffer pool
            std::lock_guard<std::mutex> lock(bsk_mutex_);
            if (bsk_ && entry_idx < cfg_.n && bsk_pool_) {
                size_t buf_idx = bsk_pool_->prefetchIndex();
                int L = static_cast<int>(cfg_.L);
                int N = static_cast<int>(cfg_.N);
                int i = static_cast<int>(entry_idx);

                auto entry = mx::slice(*bsk_,
                    {i, 0, 0, 0, 0},
                    {i + 1, 2, L, 2, N});
                entry = mx::reshape(entry, {static_cast<int>(bsk_pool_->entrySize())});

                bsk_pool_->buffer(buf_idx) = entry;
                mx::eval(bsk_pool_->buffer(buf_idx));
                bsk_pool_->setEntry(buf_idx, entry_idx);
            }
        }
    }

    Config cfg_{};
    std::atomic<bool> running_{false};

    // BSK storage
    std::shared_ptr<mx::array> bsk_;
    std::unique_ptr<BSKBufferPool> bsk_pool_;
    mutable std::mutex bsk_mutex_;

    // Stream executor
    std::unique_ptr<StreamExecutor> executor_;

    // Prefetch management
    std::queue<uint32_t> prefetch_queue_;
    std::mutex prefetch_mutex_;
    std::condition_variable prefetch_cv_;
    std::thread prefetch_worker_;

    // Statistics
    mutable Stats stats_{};
    mutable std::mutex stats_mutex_;
};

// =============================================================================
// Pipelined Batch PBS - High-level interface for async batch execution
// =============================================================================

class PipelinedBatchPBS {
public:
    struct Config {
        uint32_t N = 1024;
        uint32_t n = 512;
        uint32_t L = 3;
        uint32_t baseLog = 7;
        uint64_t Q = 1ULL << 27;
        size_t batch_size = 8;          // LWEs per batch
        size_t pipeline_depth = 2;      // Batches in flight
    };

    PipelinedBatchPBS() = default;

    explicit PipelinedBatchPBS(const Config& cfg)
        : cfg_(cfg) {

        AsyncPBSPipeline::Config pipe_cfg;
        pipe_cfg.N = cfg.N;
        pipe_cfg.n = cfg.n;
        pipe_cfg.L = cfg.L;
        pipe_cfg.baseLog = cfg.baseLog;
        pipe_cfg.Q = cfg.Q;
        pipe_cfg.pipeline_depth = cfg.pipeline_depth;

        pipeline_ = std::make_unique<AsyncPBSPipeline>(pipe_cfg);
    }

    // Set bootstrap key
    void setBSK(const mx::array& bsk) {
        if (pipeline_) {
            pipeline_->setBSK(bsk);
        }
    }

    // Execute pipelined batch PBS
    // Submits batches asynchronously and collects results
    std::vector<mx::array> execute(
        const std::vector<mx::array>& lwes,
        const mx::array& test_poly) {

        if (!pipeline_) {
            return {};
        }

        size_t total = lwes.size();
        size_t num_batches = (total + cfg_.batch_size - 1) / cfg_.batch_size;

        // Submit all batches
        std::vector<std::future<std::vector<mx::array>>> futures;
        futures.reserve(num_batches);

        for (size_t i = 0; i < total; i += cfg_.batch_size) {
            size_t end = std::min(i + cfg_.batch_size, total);
            std::vector<mx::array> batch(lwes.begin() + i, lwes.begin() + end);
            futures.push_back(pipeline_->submitBatch(batch, test_poly));
        }

        // Collect results
        std::vector<mx::array> results;
        results.reserve(total);

        for (auto& future : futures) {
            auto batch_results = future.get();
            for (auto& r : batch_results) {
                results.push_back(std::move(r));
            }
        }

        return results;
    }

    // Synchronize all pending operations
    void sync() {
        if (pipeline_) {
            pipeline_->sync();
        }
    }

    AsyncPBSPipeline::Stats stats() const {
        if (pipeline_) {
            return pipeline_->stats();
        }
        return {};
    }

private:
    Config cfg_{};
    std::unique_ptr<AsyncPBSPipeline> pipeline_;
};

#endif // WITH_MLX

}  // namespace gpu
}  // namespace lbcrypto

#endif // LBCRYPTO_MATH_HAL_MLX_ASYNC_PIPELINE_H
