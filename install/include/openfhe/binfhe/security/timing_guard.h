// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// Timing Guard - RAII guards and utilities for timing-attack resistance
//
// Provides:
// 1. RAII timing guards that ensure uniform execution time
// 2. Memory access pattern normalization
// 3. Cache flush utilities
// 4. Execution time padding to fixed durations
//
// These guards are essential for FHE operations where the ciphertext
// structure must not leak information through timing side channels.

#ifndef LUX_FHE_SECURITY_TIMING_GUARD_H
#define LUX_FHE_SECURITY_TIMING_GUARD_H

#include "constant_time.h"
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

// Platform-specific cache control
#if defined(__x86_64__) || defined(_M_X64)
    #include <x86intrin.h>
    #define LUX_HAS_CACHE_CONTROL 1
    #define LUX_CACHE_LINE_SIZE 64
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define LUX_HAS_CACHE_CONTROL 1
    #define LUX_CACHE_LINE_SIZE 64
#else
    #define LUX_HAS_CACHE_CONTROL 0
    #define LUX_CACHE_LINE_SIZE 64
#endif

// High-resolution timing
#if defined(__x86_64__) || defined(_M_X64)
    #define LUX_RDTSC() __rdtsc()
#else
    // Fallback to chrono
    #define LUX_RDTSC() (std::chrono::high_resolution_clock::now().time_since_epoch().count())
#endif

namespace lbcrypto {
namespace security {

// ============================================================================
// Cache Control Utilities
// ============================================================================

/**
 * @brief Flush a memory region from CPU cache
 *
 * Forces subsequent accesses to go to main memory.
 * Useful for ensuring cache-timing attacks can't observe access patterns.
 *
 * @param ptr Start of memory region
 * @param size Size in bytes
 */
inline void FlushCache(const void* ptr, size_t size) {
#if LUX_HAS_CACHE_CONTROL
    const char* p = static_cast<const char*>(ptr);
    const char* end = p + size;

    for (; p < end; p += LUX_CACHE_LINE_SIZE) {
    #if defined(__x86_64__) || defined(_M_X64)
        _mm_clflush(p);
    #elif defined(__aarch64__)
        // ARM DC CIVAC - Clean and Invalidate by VA to PoC
        __asm__ __volatile__("dc civac, %0" : : "r"(p) : "memory");
    #endif
    }

    // Memory barrier to ensure flushes complete
    #if defined(__x86_64__) || defined(_M_X64)
        _mm_mfence();
    #elif defined(__aarch64__)
        __asm__ __volatile__("dsb sy" ::: "memory");
    #endif
#else
    (void)ptr;
    (void)size;
#endif
}

/**
 * @brief Touch all cache lines in a memory region
 *
 * Brings data into cache uniformly, preventing timing variation
 * based on cache state.
 *
 * @param ptr Start of memory region
 * @param size Size in bytes
 */
inline void WarmCache(const void* ptr, size_t size) {
    const volatile char* p = static_cast<const volatile char*>(ptr);
    const char* end = static_cast<const char*>(ptr) + size;

    CT_BARRIER();

    volatile char sink = 0;
    for (; p < end; p += LUX_CACHE_LINE_SIZE) {
        sink ^= *p;
    }

    CT_BARRIER();
    (void)sink;  // Prevent optimization
}

/**
 * @brief Prefetch memory for upcoming read
 */
inline void PrefetchRead(const void* ptr) {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(ptr, 0, 3);  // Read, high locality
#else
    (void)ptr;
#endif
}

/**
 * @brief Prefetch memory for upcoming write
 */
inline void PrefetchWrite(void* ptr) {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(ptr, 1, 3);  // Write, high locality
#else
    (void)ptr;
#endif
}

// ============================================================================
// Timing Measurement
// ============================================================================

/**
 * @brief High-resolution timestamp counter
 *
 * Returns a monotonic timestamp suitable for measuring execution time.
 * Uses RDTSC on x86 for cycle-accurate timing.
 */
inline uint64_t GetTimestamp() {
    return LUX_RDTSC();
}

/**
 * @brief Get approximate CPU frequency for time conversion
 *
 * Estimates cycles per nanosecond by measuring a known delay.
 * Cached after first call.
 */
inline double GetCyclesPerNanosecond() {
    static double cached_freq = 0.0;
    static std::atomic<bool> initialized{false};

    if (initialized.load(std::memory_order_acquire)) {
        return cached_freq;
    }

    // Measure cycles over a known duration
    auto start_time = std::chrono::high_resolution_clock::now();
    uint64_t start_cycles = GetTimestamp();

    // Busy-wait for approximately 10ms
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    uint64_t end_cycles = GetTimestamp();
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_time - start_time).count();

    if (duration_ns > 0) {
        cached_freq = static_cast<double>(end_cycles - start_cycles) / duration_ns;
    } else {
        cached_freq = 3.0;  // Fallback: assume 3 GHz
    }

    initialized.store(true, std::memory_order_release);
    return cached_freq;
}

// ============================================================================
// RAII Timing Guards
// ============================================================================

/**
 * @brief RAII guard that ensures minimum execution time
 *
 * On destruction, busy-waits until a minimum time has elapsed
 * since construction. This prevents timing attacks that measure
 * how quickly an operation completes.
 *
 * Usage:
 *   {
 *       TimingGuard guard(1000);  // Minimum 1000 cycles
 *       // ... security-sensitive code ...
 *   }  // guard ensures minimum time elapsed
 */
class TimingGuard {
public:
    /**
     * @brief Construct guard with minimum cycle count
     *
     * @param min_cycles Minimum cycles that must elapse before destruction
     */
    explicit TimingGuard(uint64_t min_cycles)
        : start_time_(GetTimestamp())
        , min_cycles_(min_cycles)
        , enabled_(true) {}

    /**
     * @brief Construct guard with minimum nanoseconds
     *
     * @param min_ns Minimum nanoseconds
     * @param tag Unused, disambiguates from cycle constructor
     */
    TimingGuard(uint64_t min_ns, bool /* use_nanoseconds */)
        : start_time_(GetTimestamp())
        , min_cycles_(static_cast<uint64_t>(min_ns * GetCyclesPerNanosecond()))
        , enabled_(true) {}

    /**
     * @brief Destructor - enforces minimum timing
     */
    ~TimingGuard() {
        if (!enabled_) return;

        uint64_t target = start_time_ + min_cycles_;

        // Busy-wait until minimum time elapsed
        // Use volatile to prevent optimization
        volatile uint64_t now;
        do {
            CT_BARRIER();
            now = GetTimestamp();
        } while (now < target);

        CT_BARRIER();
    }

    // Non-copyable
    TimingGuard(const TimingGuard&) = delete;
    TimingGuard& operator=(const TimingGuard&) = delete;

    // Movable (transfers ownership)
    TimingGuard(TimingGuard&& other) noexcept
        : start_time_(other.start_time_)
        , min_cycles_(other.min_cycles_)
        , enabled_(other.enabled_) {
        other.enabled_ = false;
    }

    TimingGuard& operator=(TimingGuard&& other) noexcept {
        if (this != &other) {
            start_time_ = other.start_time_;
            min_cycles_ = other.min_cycles_;
            enabled_ = other.enabled_;
            other.enabled_ = false;
        }
        return *this;
    }

    /**
     * @brief Disable the guard (no timing enforcement on destruction)
     */
    void Disable() { enabled_ = false; }

    /**
     * @brief Get elapsed cycles since construction
     */
    uint64_t ElapsedCycles() const {
        return GetTimestamp() - start_time_;
    }

private:
    uint64_t start_time_;
    uint64_t min_cycles_;
    bool enabled_;
};

/**
 * @brief RAII guard that normalizes memory access patterns
 *
 * Ensures that a set of memory regions are accessed uniformly,
 * regardless of which data is actually needed. Prevents cache-timing
 * attacks based on memory access patterns.
 *
 * Usage:
 *   {
 *       MemoryAccessGuard guard;
 *       guard.AddRegion(table1, sizeof(table1));
 *       guard.AddRegion(table2, sizeof(table2));
 *       // ... code that accesses some of these regions ...
 *   }  // guard touches all regions uniformly
 */
class MemoryAccessGuard {
public:
    MemoryAccessGuard() = default;

    /**
     * @brief Destructor - touches all registered regions
     */
    ~MemoryAccessGuard() {
        if (regions_.empty()) return;

        CT_BARRIER();

        // Touch all regions to normalize access patterns
        volatile uint8_t sink = 0;
        for (const auto& region : regions_) {
            const volatile uint8_t* p = static_cast<const volatile uint8_t*>(region.ptr);
            for (size_t i = 0; i < region.size; i += LUX_CACHE_LINE_SIZE) {
                sink ^= p[i];
            }
        }

        CT_BARRIER();
        (void)sink;
    }

    /**
     * @brief Register a memory region to be touched on destruction
     */
    void AddRegion(const void* ptr, size_t size) {
        regions_.push_back({ptr, size});
    }

    /**
     * @brief Clear all registered regions
     */
    void Clear() {
        regions_.clear();
    }

    // Non-copyable
    MemoryAccessGuard(const MemoryAccessGuard&) = delete;
    MemoryAccessGuard& operator=(const MemoryAccessGuard&) = delete;

private:
    struct Region {
        const void* ptr;
        size_t size;
    };
    std::vector<Region> regions_;
};

/**
 * @brief RAII guard that combines timing and memory access protection
 *
 * Provides comprehensive side-channel protection:
 * 1. Minimum execution time
 * 2. Uniform memory access patterns
 * 3. Cache warming before and flushing after
 */
class SecureOperationGuard {
public:
    /**
     * @brief Construct with minimum execution time in cycles
     */
    explicit SecureOperationGuard(uint64_t min_cycles)
        : timing_guard_(min_cycles)
        , warm_cache_(false)
        , flush_cache_(false) {}

    ~SecureOperationGuard() {
        // Memory access guard destructor handles touching regions

        // Optionally flush cache
        if (flush_cache_) {
            for (const auto& region : regions_) {
                FlushCache(region.ptr, region.size);
            }
        }

        // Timing guard destructor handles minimum time
    }

    /**
     * @brief Add a memory region to protect
     */
    void AddRegion(const void* ptr, size_t size) {
        regions_.push_back({ptr, size});
        memory_guard_.AddRegion(ptr, size);

        if (warm_cache_) {
            WarmCache(ptr, size);
        }
    }

    /**
     * @brief Enable cache warming on region addition
     */
    void EnableCacheWarming(bool enable = true) { warm_cache_ = enable; }

    /**
     * @brief Enable cache flushing on destruction
     */
    void EnableCacheFlushing(bool enable = true) { flush_cache_ = enable; }

    // Non-copyable
    SecureOperationGuard(const SecureOperationGuard&) = delete;
    SecureOperationGuard& operator=(const SecureOperationGuard&) = delete;

private:
    struct Region {
        const void* ptr;
        size_t size;
    };

    TimingGuard timing_guard_;
    MemoryAccessGuard memory_guard_;
    std::vector<Region> regions_;
    bool warm_cache_;
    bool flush_cache_;
};

// ============================================================================
// Execution Time Padding
// ============================================================================

/**
 * @brief Execute a function with fixed execution time
 *
 * Runs the function, then pads the remaining time with busy-wait
 * to ensure constant total execution time regardless of input.
 *
 * @tparam F Function type
 * @tparam Args Argument types
 * @param min_cycles Minimum execution time in cycles
 * @param f Function to execute
 * @param args Arguments to function
 * @return Function result
 */
template<typename F, typename... Args>
auto ExecuteWithPadding(uint64_t min_cycles, F&& f, Args&&... args)
    -> decltype(f(std::forward<Args>(args)...)) {

    TimingGuard guard(min_cycles);
    return f(std::forward<Args>(args)...);
}

/**
 * @brief Execute a void function with fixed execution time
 */
template<typename F, typename... Args>
void ExecuteVoidWithPadding(uint64_t min_cycles, F&& f, Args&&... args) {
    TimingGuard guard(min_cycles);
    f(std::forward<Args>(args)...);
}

// ============================================================================
// Constant-Time Control Flow Helpers
// ============================================================================

/**
 * @brief Execute one of two functions based on condition (constant-time)
 *
 * BOTH functions are always executed to prevent timing leaks.
 * Only the result from the selected function is returned.
 *
 * @param condition Selection condition
 * @param if_true Function to use result from if condition is true
 * @param if_false Function to use result from if condition is false
 * @return Result from selected function
 */
template<typename T, typename FTrue, typename FFalse>
T ct_branch(bool condition, FTrue&& if_true, FFalse&& if_false) {
    CT_BARRIER();

    // Execute BOTH branches
    T result_true = if_true();
    T result_false = if_false();

    // Select result in constant time
    T result = ct_select_bool(condition, result_true, result_false);

    CT_BARRIER();
    return result;
}

/**
 * @brief Execute all functions, return selected result (constant-time)
 *
 * For switch-case style control flow where all cases must execute.
 *
 * @param index Which result to return
 * @param funcs Vector of functions to execute
 * @return Result from func[index]
 */
template<typename T>
T ct_switch(size_t index, const std::vector<std::function<T()>>& funcs) {
    CT_BARRIER();

    // Execute all functions
    std::vector<T> results;
    results.reserve(funcs.size());
    for (const auto& f : funcs) {
        results.push_back(f());
    }

    // Select result in constant time
    T result = ct_lookup(results.data(), results.size(), index);

    CT_BARRIER();
    return result;
}

// ============================================================================
// Loop Iteration Guards
// ============================================================================

/**
 * @brief Ensures a loop executes a fixed number of iterations
 *
 * Even if early termination condition is met, continues iterating
 * (with dummy operations) to prevent timing leaks.
 *
 * Usage:
 *   FixedIterationLoop loop(100);
 *   for (size_t i = 0; i < 100; ++i) {
 *       bool should_skip = check_condition();
 *       loop.SetSkip(should_skip);
 *
 *       if (!loop.ShouldSkip()) {
 *           // Actual work (only affects result if not skipping)
 *       }
 *       // Dummy work happens regardless
 *   }
 */
class FixedIterationLoop {
public:
    explicit FixedIterationLoop(size_t num_iterations)
        : total_iterations_(num_iterations)
        , current_iteration_(0)
        , skip_flag_(0) {}

    /**
     * @brief Set whether current iteration should be logically skipped
     *
     * The iteration still executes, but ShouldSkip() returns true.
     */
    void SetSkip(bool skip) {
        skip_flag_ = skip ? static_cast<uint8_t>(~0) : 0;
    }

    /**
     * @brief Check if current iteration should be skipped
     */
    bool ShouldSkip() const { return skip_flag_ != 0; }

    /**
     * @brief Get skip mask for constant-time selection
     */
    uint8_t GetSkipMask() const { return skip_flag_; }

    /**
     * @brief Advance to next iteration
     */
    void Next() { ++current_iteration_; }

    /**
     * @brief Check if all iterations complete
     */
    bool Done() const { return current_iteration_ >= total_iterations_; }

private:
    size_t total_iterations_;
    size_t current_iteration_;
    uint8_t skip_flag_;
};

// ============================================================================
// Secret Data Wrapper
// ============================================================================

/**
 * @brief Wrapper for secret values that enforces constant-time access
 *
 * Prevents accidental use of secret values in non-constant-time operations.
 *
 * @tparam T Underlying value type (must be integral)
 */
template<typename T>
class Secret {
    static_assert(std::is_integral<T>::value, "Secret requires integral type");

public:
    explicit Secret(T value) : value_(value) {}

    // No implicit conversion to prevent accidental leaks
    // Must use explicit constant-time operations

    /**
     * @brief Constant-time equality comparison
     */
    T Equals(T other) const {
        return ct_eq(value_, static_cast<typename std::make_unsigned<T>::type>(other));
    }

    /**
     * @brief Constant-time less-than comparison
     */
    T LessThan(T other) const {
        return ct_lt(static_cast<typename std::make_unsigned<T>::type>(value_),
                     static_cast<typename std::make_unsigned<T>::type>(other));
    }

    /**
     * @brief Constant-time select between two values based on this secret
     */
    T Select(T if_nonzero, T if_zero) const {
        auto mask = ct_is_nonzero(static_cast<typename std::make_unsigned<T>::type>(value_));
        return ct_select(static_cast<T>(mask), if_nonzero, if_zero);
    }

    /**
     * @brief Get raw value (use with caution - only in constant-time code)
     */
    T Declassify() const {
        CT_BARRIER();
        return value_;
    }

private:
    T value_;
};

} // namespace security
} // namespace lbcrypto

#endif // LUX_FHE_SECURITY_TIMING_GUARD_H
