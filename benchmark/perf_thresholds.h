// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// Performance Thresholds for FHE Operations
// Prevents performance regressions via hard limits on critical paths.
//
// Usage:
//   1. Define thresholds for each operation
//   2. Run benchmarks and compare against thresholds
//   3. Fail CI if any threshold is exceeded
//
// Thresholds are calibrated for:
//   - Apple M1/M2/M3 (baseline)
//   - Intel Xeon (server)
//   - NVIDIA A100 (GPU with MLX)

#ifndef BENCHMARK_PERF_THRESHOLDS_H
#define BENCHMARK_PERF_THRESHOLDS_H

#include <chrono>
#include <string>
#include <map>
#include <vector>
#include <cmath>
#include <iostream>
#include <sstream>
#include <functional>
#include <iomanip>

namespace lbcrypto {
namespace perf {

// ============================================================================
// Threshold Categories
// ============================================================================

enum class HardwareTarget {
    APPLE_SILICON,      // M1/M2/M3 baseline
    INTEL_XEON,         // x86-64 server
    NVIDIA_A100,        // GPU via MLX
    GENERIC_CPU         // Fallback
};

enum class ThresholdType {
    LATENCY_MS,         // Single operation latency
    THROUGHPUT_OPS,     // Operations per second
    MEMORY_MB,          // Peak memory usage
    BATCH_EFFICIENCY    // Scaling efficiency percentage
};

// ============================================================================
// Performance Threshold Definition
// ============================================================================

struct PerfThreshold {
    std::string name;           // Operation name
    ThresholdType type;         // Threshold type
    double limit;               // Threshold value
    double tolerance_pct;       // Acceptable variance (default 10%)
    std::string description;    // Human-readable description

    // Check if measured value passes threshold
    bool check(double measured) const {
        double upper_bound = limit * (1.0 + tolerance_pct / 100.0);
        switch (type) {
            case ThresholdType::LATENCY_MS:
            case ThresholdType::MEMORY_MB:
                return measured <= upper_bound;
            case ThresholdType::THROUGHPUT_OPS:
            case ThresholdType::BATCH_EFFICIENCY:
                return measured >= limit * (1.0 - tolerance_pct / 100.0);
        }
        return false;
    }

    std::string format_result(double measured, bool passed) const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3);
        ss << name << ": " << measured;
        switch (type) {
            case ThresholdType::LATENCY_MS: ss << " ms"; break;
            case ThresholdType::THROUGHPUT_OPS: ss << " ops/s"; break;
            case ThresholdType::MEMORY_MB: ss << " MB"; break;
            case ThresholdType::BATCH_EFFICIENCY: ss << "%"; break;
        }
        ss << " (limit: " << limit << ") [" << (passed ? "PASS" : "FAIL") << "]";
        return ss.str();
    }
};

// ============================================================================
// Hard Thresholds for Critical FHE Operations
// ============================================================================
// These thresholds are calibrated on Apple M2 Pro.
// Adjust for other hardware via scaling factors.

namespace thresholds {

// NTT Thresholds (milliseconds)
// Ring dimension N=4096 is the critical path for fhEVM
inline const PerfThreshold NTT_4096_BATCH_1 = {
    "NTT-4096/batch=1", ThresholdType::LATENCY_MS, 0.8, 15.0,
    "Single NTT-4096 forward transform"
};

inline const PerfThreshold NTT_4096_BATCH_8 = {
    "NTT-4096/batch=8", ThresholdType::LATENCY_MS, 2.0, 15.0,
    "Batch NTT-4096 forward transform (8 polynomials)"
};

inline const PerfThreshold NTT_4096_BATCH_32 = {
    "NTT-4096/batch=32", ThresholdType::LATENCY_MS, 5.0, 15.0,
    "Batch NTT-4096 forward transform (32 polynomials)"
};

inline const PerfThreshold NTT_4096_BATCH_128 = {
    "NTT-4096/batch=128", ThresholdType::LATENCY_MS, 15.0, 15.0,
    "Batch NTT-4096 forward transform (128 polynomials)"
};

// INTT Thresholds (slightly higher than forward due to scaling)
inline const PerfThreshold INTT_4096_BATCH_32 = {
    "INTT-4096/batch=32", ThresholdType::LATENCY_MS, 6.0, 15.0,
    "Batch INTT-4096 inverse transform (32 polynomials)"
};

// Barrett Multiplication Kernel
inline const PerfThreshold BARRETT_MUL_THROUGHPUT = {
    "BarrettMul/throughput", ThresholdType::THROUGHPUT_OPS, 1e9, 10.0,
    "Barrett modular multiplication throughput (ops/sec)"
};

inline const PerfThreshold BARRETT_MUL_BATCH_1K = {
    "BarrettMul/batch=1000", ThresholdType::LATENCY_MS, 0.1, 20.0,
    "Batch Barrett mul for 1000 elements"
};

// External Product (key operation in bootstrapping)
inline const PerfThreshold EXT_PRODUCT_SINGLE = {
    "ExtProduct/single", ThresholdType::LATENCY_MS, 8.0, 15.0,
    "Single external product operation"
};

inline const PerfThreshold EXT_PRODUCT_BATCH_8 = {
    "ExtProduct/batch=8", ThresholdType::LATENCY_MS, 30.0, 15.0,
    "Batch external product (8 operations)"
};

// Programmable Bootstrapping (PBS)
inline const PerfThreshold PBS_SINGLE = {
    "PBS/single", ThresholdType::LATENCY_MS, 15.0, 15.0,
    "Single programmable bootstrapping"
};

inline const PerfThreshold PBS_BATCH_32 = {
    "PBS/batch=32", ThresholdType::LATENCY_MS, 100.0, 20.0,
    "Batch PBS (32 ciphertexts)"
};

// Carry Propagation (normalize)
inline const PerfThreshold NORMALIZE_DEPTH_4 = {
    "Normalize/depth=4", ThresholdType::LATENCY_MS, 50.0, 15.0,
    "Carry propagation with depth 4 (8-bit radix)"
};

inline const PerfThreshold NORMALIZE_DEPTH_8 = {
    "Normalize/depth=8", ThresholdType::LATENCY_MS, 100.0, 15.0,
    "Carry propagation with depth 8 (64-bit radix)"
};

inline const PerfThreshold NORMALIZE_DEPTH_128 = {
    "Normalize/depth=128", ThresholdType::LATENCY_MS, 1000.0, 20.0,
    "Carry propagation with depth 128 (256-bit radix)"
};

// Comparison Operations
inline const PerfThreshold CMP_LT_8BIT = {
    "Lt/8bit", ThresholdType::LATENCY_MS, 40.0, 15.0,
    "Less-than comparison on 8-bit radix"
};

inline const PerfThreshold CMP_LT_64BIT = {
    "Lt/64bit", ThresholdType::LATENCY_MS, 300.0, 15.0,
    "Less-than comparison on 64-bit radix"
};

inline const PerfThreshold CMP_LT_256BIT = {
    "Lt/256bit", ThresholdType::LATENCY_MS, 1200.0, 20.0,
    "Less-than comparison on 256-bit radix (EVM uint256)"
};

inline const PerfThreshold CMP_EQ_256BIT = {
    "Eq/256bit", ThresholdType::LATENCY_MS, 200.0, 15.0,
    "Equality comparison on 256-bit radix"
};

// Shift Operations
inline const PerfThreshold SHIFT_BYTE_ALIGNED = {
    "Shl/byte-aligned", ThresholdType::LATENCY_MS, 5.0, 15.0,
    "Left shift by multiple of 8 bits (cheap)"
};

inline const PerfThreshold SHIFT_GENERAL = {
    "Shl/general", ThresholdType::LATENCY_MS, 50.0, 15.0,
    "General left shift by encrypted amount"
};

inline const PerfThreshold SHIFT_ENCRYPTED_256 = {
    "Shl/encrypted-256bit", ThresholdType::LATENCY_MS, 400.0, 20.0,
    "Shift uint256 by encrypted amount"
};

// Radix Arithmetic
inline const PerfThreshold RADIX_ADD_256BIT = {
    "RadixAdd/256bit", ThresholdType::LATENCY_MS, 150.0, 15.0,
    "Addition of two 256-bit radix integers"
};

inline const PerfThreshold RADIX_MUL_64BIT = {
    "RadixMul/64bit", ThresholdType::LATENCY_MS, 500.0, 20.0,
    "Multiplication of two 64-bit radix integers"
};

inline const PerfThreshold RADIX_MUL_256BIT = {
    "RadixMul/256bit", ThresholdType::LATENCY_MS, 8000.0, 25.0,
    "Multiplication of two 256-bit radix integers (heavy)"
};

// Batch Efficiency (percentage of linear scaling)
inline const PerfThreshold BATCH_EFFICIENCY_NTT = {
    "BatchEff/NTT", ThresholdType::BATCH_EFFICIENCY, 80.0, 5.0,
    "NTT batch efficiency (should scale near-linearly)"
};

inline const PerfThreshold BATCH_EFFICIENCY_PBS = {
    "BatchEff/PBS", ThresholdType::BATCH_EFFICIENCY, 70.0, 10.0,
    "PBS batch efficiency"
};

} // namespace thresholds

// ============================================================================
// Hardware Scaling Factors
// ============================================================================

inline double get_hardware_scaling(HardwareTarget target) {
    switch (target) {
        case HardwareTarget::APPLE_SILICON: return 1.0;   // Baseline
        case HardwareTarget::INTEL_XEON:    return 1.3;   // ~30% slower
        case HardwareTarget::NVIDIA_A100:   return 0.2;   // 5x faster with GPU
        case HardwareTarget::GENERIC_CPU:   return 2.0;   // Conservative
    }
    return 2.0;
}

inline HardwareTarget detect_hardware() {
#ifdef __APPLE__
    #if defined(__arm64__) || defined(__aarch64__)
        return HardwareTarget::APPLE_SILICON;
    #else
        return HardwareTarget::GENERIC_CPU;
    #endif
#elif defined(__linux__)
    #ifdef WITH_MLX
        return HardwareTarget::NVIDIA_A100;
    #else
        return HardwareTarget::INTEL_XEON;
    #endif
#else
    return HardwareTarget::GENERIC_CPU;
#endif
}

// ============================================================================
// Benchmark Result
// ============================================================================

struct BenchmarkResult {
    std::string name;
    double measured;
    double threshold;
    bool passed;
    std::string unit;
    int iterations;
    double std_dev;

    std::string to_json() const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(6);
        ss << "  {\"name\": \"" << name << "\", "
           << "\"measured\": " << measured << ", "
           << "\"threshold\": " << threshold << ", "
           << "\"passed\": " << (passed ? "true" : "false") << ", "
           << "\"unit\": \"" << unit << "\", "
           << "\"iterations\": " << iterations << ", "
           << "\"std_dev\": " << std_dev << "}";
        return ss.str();
    }
};

// ============================================================================
// Benchmark Runner with Threshold Checking
// ============================================================================

class ThresholdBenchmark {
public:
    ThresholdBenchmark() : hardware_(detect_hardware()) {
        scaling_ = get_hardware_scaling(hardware_);
    }

    // Run benchmark and check against threshold
    template<typename Func>
    BenchmarkResult run(const PerfThreshold& threshold, Func&& func,
                        int iterations = 100, int warmup = 5) {
        // Warmup
        for (int i = 0; i < warmup; i++) {
            func();
        }

        // Collect timing samples
        std::vector<double> samples;
        samples.reserve(iterations);

        for (int i = 0; i < iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            func();
            auto end = std::chrono::high_resolution_clock::now();

            double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
            samples.push_back(elapsed);
        }

        // Calculate statistics
        double sum = 0.0;
        for (double s : samples) sum += s;
        double mean = sum / iterations;

        double var_sum = 0.0;
        for (double s : samples) {
            var_sum += (s - mean) * (s - mean);
        }
        double std_dev = std::sqrt(var_sum / iterations);

        // Apply hardware scaling to threshold
        double scaled_threshold = threshold.limit * scaling_;
        bool passed = threshold.check(mean / scaling_);

        BenchmarkResult result;
        result.name = threshold.name;
        result.measured = mean;
        result.threshold = scaled_threshold;
        result.passed = passed;
        result.iterations = iterations;
        result.std_dev = std_dev;

        switch (threshold.type) {
            case ThresholdType::LATENCY_MS: result.unit = "ms"; break;
            case ThresholdType::THROUGHPUT_OPS: result.unit = "ops/s"; break;
            case ThresholdType::MEMORY_MB: result.unit = "MB"; break;
            case ThresholdType::BATCH_EFFICIENCY: result.unit = "%"; break;
        }

        results_.push_back(result);
        return result;
    }

    // Run throughput benchmark
    template<typename Func>
    BenchmarkResult run_throughput(const PerfThreshold& threshold, Func&& func,
                                    size_t ops_per_call, int iterations = 100) {
        std::vector<double> samples;

        for (int i = 0; i < iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            func();
            auto end = std::chrono::high_resolution_clock::now();

            double elapsed_s = std::chrono::duration<double>(end - start).count();
            samples.push_back(ops_per_call / elapsed_s);
        }

        double sum = 0.0;
        for (double s : samples) sum += s;
        double mean = sum / iterations;

        double scaled_threshold = threshold.limit / scaling_;
        bool passed = (mean >= scaled_threshold);

        BenchmarkResult result;
        result.name = threshold.name;
        result.measured = mean;
        result.threshold = scaled_threshold;
        result.passed = passed;
        result.iterations = iterations;
        result.unit = "ops/s";

        results_.push_back(result);
        return result;
    }

    // Check batch scaling efficiency
    BenchmarkResult check_batch_efficiency(const PerfThreshold& threshold,
                                           double single_time, double batch_time,
                                           int batch_size) {
        double ideal_time = single_time * batch_size;
        double efficiency = (ideal_time / batch_time) * 100.0;

        bool passed = (efficiency >= threshold.limit);

        BenchmarkResult result;
        result.name = threshold.name;
        result.measured = efficiency;
        result.threshold = threshold.limit;
        result.passed = passed;
        result.iterations = 1;
        result.unit = "%";

        results_.push_back(result);
        return result;
    }

    // Get all results
    const std::vector<BenchmarkResult>& results() const { return results_; }

    // Check if all tests passed
    bool all_passed() const {
        for (const auto& r : results_) {
            if (!r.passed) return false;
        }
        return true;
    }

    // Count failures
    int failure_count() const {
        int count = 0;
        for (const auto& r : results_) {
            if (!r.passed) count++;
        }
        return count;
    }

    // Print summary
    void print_summary(std::ostream& os = std::cout) const {
        os << "\n=== Performance Benchmark Summary ===\n";
        os << "Hardware: ";
        switch (hardware_) {
            case HardwareTarget::APPLE_SILICON: os << "Apple Silicon"; break;
            case HardwareTarget::INTEL_XEON: os << "Intel Xeon"; break;
            case HardwareTarget::NVIDIA_A100: os << "NVIDIA A100 (MLX)"; break;
            case HardwareTarget::GENERIC_CPU: os << "Generic CPU"; break;
        }
        os << " (scaling: " << scaling_ << "x)\n\n";

        for (const auto& r : results_) {
            os << std::left << std::setw(35) << r.name
               << std::right << std::setw(12) << std::fixed << std::setprecision(3) << r.measured
               << " " << std::left << std::setw(6) << r.unit
               << " (limit: " << std::setw(10) << r.threshold << ")"
               << " [" << (r.passed ? "PASS" : "FAIL") << "]\n";
        }

        os << "\n" << results_.size() << " benchmarks, "
           << failure_count() << " failures\n";
    }

    // Output JSON for CI integration
    std::string to_json() const {
        std::ostringstream ss;
        ss << "{\n";
        ss << "\"hardware\": \"" << static_cast<int>(hardware_) << "\",\n";
        ss << "\"scaling\": " << scaling_ << ",\n";
        ss << "\"results\": [\n";

        for (size_t i = 0; i < results_.size(); i++) {
            ss << results_[i].to_json();
            if (i < results_.size() - 1) ss << ",";
            ss << "\n";
        }

        ss << "],\n";
        ss << "\"total\": " << results_.size() << ",\n";
        ss << "\"failures\": " << failure_count() << ",\n";
        ss << "\"passed\": " << (all_passed() ? "true" : "false") << "\n";
        ss << "}\n";

        return ss.str();
    }

private:
    HardwareTarget hardware_;
    double scaling_;
    std::vector<BenchmarkResult> results_;
};

// ============================================================================
// Macros for Convenient Test Definition
// ============================================================================

#define BENCHMARK_WITH_THRESHOLD(bench, threshold, func)               \
    do {                                                               \
        auto result = (bench).run((threshold), [&]() { func; });       \
        EXPECT_TRUE(result.passed) << result.name << " exceeded limit: " \
            << result.measured << " " << result.unit                   \
            << " > " << result.threshold << " " << result.unit;        \
    } while(0)

#define BENCHMARK_THROUGHPUT(bench, threshold, ops_per_call, func)     \
    do {                                                               \
        auto result = (bench).run_throughput(                          \
            (threshold), [&]() { func; }, (ops_per_call));             \
        EXPECT_TRUE(result.passed) << result.name << " below limit: "  \
            << result.measured << " ops/s < "                          \
            << result.threshold << " ops/s";                           \
    } while(0)

} // namespace perf
} // namespace lbcrypto

#endif // BENCHMARK_PERF_THRESHOLDS_H
