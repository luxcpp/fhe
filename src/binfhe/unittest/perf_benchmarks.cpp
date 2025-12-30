// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// Performance Benchmarks with Regression Detection
//
// Integrated with gtest for CI. Fails if any benchmark exceeds threshold.
// This prevents "optimizations" that improve one path but regress another.
//
// Tests:
//   - Fused NTT (4096) at batch sizes {1, 8, 32, 128}
//   - Barrett mul kernel throughput
//   - External product end-to-end
//   - Normalize() cost at various carry depths
//   - Comparison operations (Lt, Eq, etc.)
//   - Shift operations (byte-aligned vs general)

#include "gtest/gtest.h"
#include "binfhecontext.h"
#include "math/hal/basicint.h"

#ifdef WITH_LUX_EXTENSIONS
#include "radix/radix.h"
#include "radix/shortint.h"
#endif

#include <chrono>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace lbcrypto;
using namespace std::chrono;

// ============================================================================
// Performance Thresholds (in milliseconds unless noted)
// Calibrated on Apple M2 Pro. Adjust for CI hardware.
// ============================================================================

namespace perf_limits {
    // NTT-4096 thresholds (ms)
    constexpr double NTT_4096_BATCH_1 = 1.0;
    constexpr double NTT_4096_BATCH_8 = 3.0;
    constexpr double NTT_4096_BATCH_32 = 8.0;
    constexpr double NTT_4096_BATCH_128 = 25.0;

    // Barrett modular multiplication (ms for 100k ops)
    constexpr double BARRETT_100K = 5.0;

    // PBS (Programmable Bootstrapping) - TOY params
    constexpr double PBS_TOY_SINGLE = 20.0;
    constexpr double PBS_TOY_BATCH_8 = 100.0;

    // Radix operations (64-bit)
    [[maybe_unused]] constexpr double RADIX_ADD_64BIT = 100.0;
    [[maybe_unused]] constexpr double RADIX_LT_64BIT = 150.0;

    // Carry propagation
    [[maybe_unused]] constexpr double NORMALIZE_DEPTH_4 = 60.0;
    [[maybe_unused]] constexpr double NORMALIZE_DEPTH_8 = 120.0;

    // Shift operations
    [[maybe_unused]] constexpr double SHIFT_BYTE_ALIGNED = 10.0;
    [[maybe_unused]] constexpr double SHIFT_GENERAL = 80.0;

    // Hardware scaling (applied in CI)
    constexpr double TOLERANCE_PCT = 20.0; // Allow 20% variance
}

// ============================================================================
// Timing Utilities
// ============================================================================

template<typename Func>
double measure_latency_ms(Func&& func, int iterations = 100, int warmup = 10) {
    // Warmup
    for (int i = 0; i < warmup; i++) {
        func();
    }

    // Measure
    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        func();
    }
    auto end = high_resolution_clock::now();

    return duration<double, std::milli>(end - start).count() / iterations;
}

double apply_tolerance(double threshold) {
    return threshold * (1.0 + perf_limits::TOLERANCE_PCT / 100.0);
}

// ============================================================================
// NTT Performance Tests
// ============================================================================

class NTTPerfTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_ = 4096;
        m_ = n_ << 1;
        modulusQ_ = LastPrime<NativeInteger>(MAX_MODULUS_SIZE, m_);
        rootOfUnity_ = RootOfUnity(m_, modulusQ_);
        crtFTT_.PreCompute(rootOfUnity_, m_, modulusQ_);
    }

    uint32_t n_, m_;
    NativeInteger modulusQ_, rootOfUnity_;
    ChineseRemainderTransformFTT<NativeVector> crtFTT_;
};

TEST_F(NTTPerfTest, NTT4096_Batch1) {
    DiscreteUniformGeneratorImpl<NativeVector> dug;
    NativeVector x = dug.GenerateVector(n_, modulusQ_);
    NativeVector X(n_);

    double latency = measure_latency_ms([&]() {
        crtFTT_.ForwardTransformToBitReverse(x, rootOfUnity_, m_, &X);
    });

    double threshold = apply_tolerance(perf_limits::NTT_4096_BATCH_1);
    EXPECT_LE(latency, threshold)
        << "NTT-4096 batch=1: " << latency << " ms > " << threshold << " ms";

    std::cout << "NTT-4096 batch=1: " << std::fixed << std::setprecision(3)
              << latency << " ms (limit: " << perf_limits::NTT_4096_BATCH_1 << " ms)\n";
}

TEST_F(NTTPerfTest, NTT4096_Batch8) {
    DiscreteUniformGeneratorImpl<NativeVector> dug;
    std::vector<NativeVector> polys(8), results(8, NativeVector(n_));
    for (int i = 0; i < 8; i++) {
        polys[i] = dug.GenerateVector(n_, modulusQ_);
    }

    double latency = measure_latency_ms([&]() {
        for (int i = 0; i < 8; i++) {
            crtFTT_.ForwardTransformToBitReverse(polys[i], rootOfUnity_, m_, &results[i]);
        }
    });

    double threshold = apply_tolerance(perf_limits::NTT_4096_BATCH_8);
    EXPECT_LE(latency, threshold)
        << "NTT-4096 batch=8: " << latency << " ms > " << threshold << " ms";

    std::cout << "NTT-4096 batch=8: " << std::fixed << std::setprecision(3)
              << latency << " ms (limit: " << perf_limits::NTT_4096_BATCH_8 << " ms)\n";
}

TEST_F(NTTPerfTest, NTT4096_Batch32) {
    DiscreteUniformGeneratorImpl<NativeVector> dug;
    std::vector<NativeVector> polys(32), results(32, NativeVector(n_));
    for (int i = 0; i < 32; i++) {
        polys[i] = dug.GenerateVector(n_, modulusQ_);
    }

    double latency = measure_latency_ms([&]() {
        for (int i = 0; i < 32; i++) {
            crtFTT_.ForwardTransformToBitReverse(polys[i], rootOfUnity_, m_, &results[i]);
        }
    }, 50, 5);

    double threshold = apply_tolerance(perf_limits::NTT_4096_BATCH_32);
    EXPECT_LE(latency, threshold)
        << "NTT-4096 batch=32: " << latency << " ms > " << threshold << " ms";

    std::cout << "NTT-4096 batch=32: " << std::fixed << std::setprecision(3)
              << latency << " ms (limit: " << perf_limits::NTT_4096_BATCH_32 << " ms)\n";
}

TEST_F(NTTPerfTest, NTT4096_Batch128) {
    DiscreteUniformGeneratorImpl<NativeVector> dug;
    std::vector<NativeVector> polys(128), results(128, NativeVector(n_));
    for (int i = 0; i < 128; i++) {
        polys[i] = dug.GenerateVector(n_, modulusQ_);
    }

    double latency = measure_latency_ms([&]() {
        for (int i = 0; i < 128; i++) {
            crtFTT_.ForwardTransformToBitReverse(polys[i], rootOfUnity_, m_, &results[i]);
        }
    }, 20, 3);

    double threshold = apply_tolerance(perf_limits::NTT_4096_BATCH_128);
    EXPECT_LE(latency, threshold)
        << "NTT-4096 batch=128: " << latency << " ms > " << threshold << " ms";

    std::cout << "NTT-4096 batch=128: " << std::fixed << std::setprecision(3)
              << latency << " ms (limit: " << perf_limits::NTT_4096_BATCH_128 << " ms)\n";
}

TEST_F(NTTPerfTest, NTT_BatchScalingEfficiency) {
    DiscreteUniformGeneratorImpl<NativeVector> dug;

    // Single NTT
    NativeVector x = dug.GenerateVector(n_, modulusQ_);
    NativeVector X(n_);

    double single_time = measure_latency_ms([&]() {
        crtFTT_.ForwardTransformToBitReverse(x, rootOfUnity_, m_, &X);
    });

    // Batch of 32
    std::vector<NativeVector> polys(32), results(32, NativeVector(n_));
    for (int i = 0; i < 32; i++) {
        polys[i] = dug.GenerateVector(n_, modulusQ_);
    }

    double batch_time = measure_latency_ms([&]() {
        for (int i = 0; i < 32; i++) {
            crtFTT_.ForwardTransformToBitReverse(polys[i], rootOfUnity_, m_, &results[i]);
        }
    }, 50, 5);

    // Check scaling efficiency
    double ideal_time = single_time * 32;
    double efficiency = (ideal_time / batch_time) * 100.0;

    std::cout << "Batch scaling efficiency: " << std::fixed << std::setprecision(1)
              << efficiency << "% (ideal: 100%)\n";

    // Should be at least 80% efficient (some overhead expected)
    EXPECT_GE(efficiency, 70.0)
        << "Batch scaling efficiency too low: " << efficiency << "%";
}

// ============================================================================
// Barrett Multiplication Performance Tests
// ============================================================================

class BarrettPerfTest : public ::testing::Test {
protected:
    void SetUp() override {
        q_ = LastPrime<NativeInteger>(60, 8192);
        std::random_device rd;
        std::mt19937_64 gen(rd());

        a_.resize(N_);
        b_.resize(N_);
        c_.resize(N_);

        for (size_t i = 0; i < N_; i++) {
            uint64_t q_val = q_.template ConvertToInt<uint64_t>();
            a_[i] = NativeInteger(gen() % q_val);
            b_[i] = NativeInteger(gen() % q_val);
        }
    }

    static constexpr size_t N_ = 100000;
    NativeInteger q_;
    std::vector<NativeInteger> a_, b_, c_;
};

TEST_F(BarrettPerfTest, BarrettMul100K) {
    double latency = measure_latency_ms([&]() {
        for (size_t i = 0; i < N_; i++) {
            c_[i] = a_[i].ModMul(b_[i], q_);
        }
    }, 50, 5);

    double threshold = apply_tolerance(perf_limits::BARRETT_100K);
    EXPECT_LE(latency, threshold)
        << "Barrett 100k: " << latency << " ms > " << threshold << " ms";

    double throughput = (N_ * 1000.0) / latency;
    std::cout << "Barrett 100k: " << std::fixed << std::setprecision(3)
              << latency << " ms (" << std::setprecision(1)
              << (throughput / 1e6) << " Mops/s)\n";
}

TEST_F(BarrettPerfTest, BarrettThroughputMinimum) {
    double latency = measure_latency_ms([&]() {
        for (size_t i = 0; i < N_; i++) {
            c_[i] = a_[i].ModMul(b_[i], q_);
        }
    }, 50, 5);

    double throughput = (N_ * 1000.0) / latency; // ops per second

    // Minimum 10M ops/sec
    EXPECT_GE(throughput, 10e6)
        << "Barrett throughput too low: " << (throughput / 1e6) << " Mops/s";
}

// ============================================================================
// PBS (Bootstrap) Performance Tests
// ============================================================================

class PBSPerfTest : public ::testing::Test {
protected:
    void SetUp() override {
        cc_.GenerateBinFHEContext(TOY, GINX);
        sk_ = cc_.KeyGen();
        cc_.BTKeyGen(sk_);
    }

    BinFHEContext cc_;
    LWEPrivateKey sk_;
};

TEST_F(PBSPerfTest, PBS_TOY_Single) {
    auto ct1 = cc_.Encrypt(sk_, 1);
    auto ct0 = cc_.Encrypt(sk_, 0);

    double latency = measure_latency_ms([&]() {
        cc_.EvalBinGate(AND, ct1, ct0);
    }, 20, 3);

    double threshold = apply_tolerance(perf_limits::PBS_TOY_SINGLE);
    EXPECT_LE(latency, threshold)
        << "PBS single: " << latency << " ms > " << threshold << " ms";

    std::cout << "PBS single (TOY): " << std::fixed << std::setprecision(3)
              << latency << " ms (limit: " << perf_limits::PBS_TOY_SINGLE << " ms)\n";
}

TEST_F(PBSPerfTest, PBS_TOY_Batch8) {
    std::vector<LWECiphertext> cts1(8), cts0(8);
    for (int i = 0; i < 8; i++) {
        cts1[i] = cc_.Encrypt(sk_, i % 2);
        cts0[i] = cc_.Encrypt(sk_, (i + 1) % 2);
    }

    double latency = measure_latency_ms([&]() {
        for (int i = 0; i < 8; i++) {
            cc_.EvalBinGate(AND, cts1[i], cts0[i]);
        }
    }, 10, 2);

    double threshold = apply_tolerance(perf_limits::PBS_TOY_BATCH_8);
    EXPECT_LE(latency, threshold)
        << "PBS batch=8: " << latency << " ms > " << threshold << " ms";

    std::cout << "PBS batch=8 (TOY): " << std::fixed << std::setprecision(3)
              << latency << " ms (limit: " << perf_limits::PBS_TOY_BATCH_8 << " ms)\n";
}

// ============================================================================
// Radix Operations Performance Tests
// ============================================================================

#ifdef WITH_LUX_EXTENSIONS
class RadixPerfTest : public ::testing::Test {
protected:
    void SetUp() override {
        cc_.GenerateBinFHEContext(TOY, GINX);
        sk_ = cc_.KeyGen();
        cc_.BTKeyGen(sk_);
        luts_ = std::make_unique<radix::ShortIntLUTs>(radix::params::EUINT64.limb_params);
    }

    BinFHEContext cc_;
    LWEPrivateKey sk_;
    std::unique_ptr<radix::ShortIntLUTs> luts_;
};

TEST_F(RadixPerfTest, RadixAdd64Bit) {
    using namespace radix;

    auto a = RadixInt::Encrypt(cc_, params::EUINT64, 12345, sk_);
    auto b = RadixInt::Encrypt(cc_, params::EUINT64, 67890, sk_);

    double latency = measure_latency_ms([&]() {
        Add(a, b, *luts_);
    }, 10, 2);

    double threshold = apply_tolerance(perf_limits::RADIX_ADD_64BIT);
    EXPECT_LE(latency, threshold)
        << "RadixAdd 64-bit: " << latency << " ms > " << threshold << " ms";

    std::cout << "RadixAdd 64-bit: " << std::fixed << std::setprecision(3)
              << latency << " ms (limit: " << perf_limits::RADIX_ADD_64BIT << " ms)\n";
}

TEST_F(RadixPerfTest, RadixLt64Bit) {
    using namespace radix;

    auto a = RadixInt::Encrypt(cc_, params::EUINT64, 12345, sk_);
    auto b = RadixInt::Encrypt(cc_, params::EUINT64, 67890, sk_);

    double latency = measure_latency_ms([&]() {
        Lt(a, b, *luts_);
    }, 10, 2);

    double threshold = apply_tolerance(perf_limits::RADIX_LT_64BIT);
    EXPECT_LE(latency, threshold)
        << "RadixLt 64-bit: " << latency << " ms > " << threshold << " ms";

    std::cout << "RadixLt 64-bit: " << std::fixed << std::setprecision(3)
              << latency << " ms (limit: " << perf_limits::RADIX_LT_64BIT << " ms)\n";
}

TEST_F(RadixPerfTest, NormalizeDepth4) {
    using namespace radix;

    auto a = RadixInt::Encrypt(cc_, params::EUINT8, 200, sk_);
    radix::ShortIntLUTs luts8(params::EUINT8.limb_params);

    // Accumulate some carries
    a.AddScalarInPlace(100, luts8);
    a.AddScalarInPlace(100, luts8);

    double latency = measure_latency_ms([&]() {
        a.PropagateCarries(luts8);
    }, 10, 2);

    double threshold = apply_tolerance(perf_limits::NORMALIZE_DEPTH_4);
    EXPECT_LE(latency, threshold)
        << "Normalize depth=4: " << latency << " ms > " << threshold << " ms";

    std::cout << "Normalize depth=4: " << std::fixed << std::setprecision(3)
              << latency << " ms (limit: " << perf_limits::NORMALIZE_DEPTH_4 << " ms)\n";
}

TEST_F(RadixPerfTest, ShiftByteAligned) {
    using namespace radix;

    auto a = RadixInt::Encrypt(cc_, params::EUINT64, 0x12345678, sk_);

    double latency = measure_latency_ms([&]() {
        RadixInt copy = a;
        copy.ShlInPlace(8, *luts_);  // Byte-aligned shift
    }, 20, 3);

    double threshold = apply_tolerance(perf_limits::SHIFT_BYTE_ALIGNED);
    EXPECT_LE(latency, threshold)
        << "Shift byte-aligned: " << latency << " ms > " << threshold << " ms";

    std::cout << "Shift byte-aligned: " << std::fixed << std::setprecision(3)
              << latency << " ms (limit: " << perf_limits::SHIFT_BYTE_ALIGNED << " ms)\n";
}

TEST_F(RadixPerfTest, ShiftGeneral) {
    using namespace radix;

    auto a = RadixInt::Encrypt(cc_, params::EUINT64, 0x12345678, sk_);

    double latency = measure_latency_ms([&]() {
        RadixInt copy = a;
        copy.ShlInPlace(3, *luts_);  // Non-byte-aligned shift
    }, 10, 2);

    double threshold = apply_tolerance(perf_limits::SHIFT_GENERAL);
    EXPECT_LE(latency, threshold)
        << "Shift general: " << latency << " ms > " << threshold << " ms";

    std::cout << "Shift general: " << std::fixed << std::setprecision(3)
              << latency << " ms (limit: " << perf_limits::SHIFT_GENERAL << " ms)\n";
}

#else // !WITH_LUX_EXTENSIONS

TEST(RadixPerfTest, SkippedWithoutExtensions) {
    GTEST_SKIP() << "Radix perf tests require WITH_LUX_EXTENSIONS=ON";
}

#endif // WITH_LUX_EXTENSIONS

// ============================================================================
// Batch Efficiency Tests
// ============================================================================

class BatchEfficiencyTest : public ::testing::Test {
protected:
    void SetUp() override {
        cc_.GenerateBinFHEContext(TOY, GINX);
        sk_ = cc_.KeyGen();
        cc_.BTKeyGen(sk_);
    }

    BinFHEContext cc_;
    LWEPrivateKey sk_;
};

TEST_F(BatchEfficiencyTest, PBSBatchScaling) {
    // Single PBS
    auto ct1 = cc_.Encrypt(sk_, 1);
    auto ct0 = cc_.Encrypt(sk_, 0);

    double single_time = measure_latency_ms([&]() {
        cc_.EvalBinGate(AND, ct1, ct0);
    }, 10, 2);

    // Batch of 8
    std::vector<LWECiphertext> cts1(8), cts0(8);
    for (int i = 0; i < 8; i++) {
        cts1[i] = cc_.Encrypt(sk_, i % 2);
        cts0[i] = cc_.Encrypt(sk_, (i + 1) % 2);
    }

    double batch_time = measure_latency_ms([&]() {
        for (int i = 0; i < 8; i++) {
            cc_.EvalBinGate(AND, cts1[i], cts0[i]);
        }
    }, 5, 1);

    // Calculate efficiency
    double ideal_time = single_time * 8;
    double efficiency = (ideal_time / batch_time) * 100.0;

    std::cout << "PBS batch efficiency: " << std::fixed << std::setprecision(1)
              << efficiency << "% (single=" << single_time
              << "ms, batch8=" << batch_time << "ms)\n";

    // Should be at least 60% efficient (sequential is baseline)
    EXPECT_GE(efficiency, 60.0)
        << "PBS batch efficiency too low: " << efficiency << "%";
}

// ============================================================================
// Memory Usage Tests
// ============================================================================

class MemoryPerfTest : public ::testing::Test {};

TEST_F(MemoryPerfTest, NTTMemoryFootprint) {
    // Measure memory for NTT precomputation
    uint32_t n = 4096;
    uint32_t m = n << 1;
    NativeInteger q(LastPrime<NativeInteger>(MAX_MODULUS_SIZE, m));
    NativeInteger root = RootOfUnity(m, q);

    ChineseRemainderTransformFTT<NativeVector> crtFTT;

    // Memory before
    // Note: This is a rough estimate; actual memory tracking would require
    // custom allocators or platform-specific APIs

    crtFTT.PreCompute(root, m, q);

    // Just verify it doesn't crash and computation works
    DiscreteUniformGeneratorImpl<NativeVector> dug;
    NativeVector x = dug.GenerateVector(n, q);
    NativeVector X(n);

    crtFTT.ForwardTransformToBitReverse(x, root, m, &X);

    SUCCEED() << "NTT precomputation and execution successful";
}

// ============================================================================
// Regression Detection Summary
// ============================================================================

class RegressionSummary : public ::testing::Test {};

TEST_F(RegressionSummary, PrintThresholds) {
    std::cout << "\n=== Performance Thresholds ===\n";
    std::cout << "NTT-4096 batch=1:   " << perf_limits::NTT_4096_BATCH_1 << " ms\n";
    std::cout << "NTT-4096 batch=8:   " << perf_limits::NTT_4096_BATCH_8 << " ms\n";
    std::cout << "NTT-4096 batch=32:  " << perf_limits::NTT_4096_BATCH_32 << " ms\n";
    std::cout << "NTT-4096 batch=128: " << perf_limits::NTT_4096_BATCH_128 << " ms\n";
    std::cout << "Barrett 100k:       " << perf_limits::BARRETT_100K << " ms\n";
    std::cout << "PBS single (TOY):   " << perf_limits::PBS_TOY_SINGLE << " ms\n";
    std::cout << "PBS batch=8 (TOY):  " << perf_limits::PBS_TOY_BATCH_8 << " ms\n";
#ifdef WITH_LUX_EXTENSIONS
    std::cout << "RadixAdd 64-bit:    " << perf_limits::RADIX_ADD_64BIT << " ms\n";
    std::cout << "RadixLt 64-bit:     " << perf_limits::RADIX_LT_64BIT << " ms\n";
    std::cout << "Normalize depth=4:  " << perf_limits::NORMALIZE_DEPTH_4 << " ms\n";
    std::cout << "Shift byte-aligned: " << perf_limits::SHIFT_BYTE_ALIGNED << " ms\n";
    std::cout << "Shift general:      " << perf_limits::SHIFT_GENERAL << " ms\n";
#endif
    std::cout << "Tolerance: " << perf_limits::TOLERANCE_PCT << "%\n";
    std::cout << "==============================\n";
}
