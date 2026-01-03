// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// FHE Microbenchmark Suite
// Standalone benchmark binary with JSON output for CI integration.
//
// Measures:
//   - Fused NTT (4096) at batch sizes {1, 8, 32, 128}
//   - Barrett mul kernel throughput
//   - External product end-to-end
//   - Normalize() cost at various carry depths
//   - Comparison operations (Lt, Eq, etc.)
//   - Shift operations (byte-aligned vs general)
//
// Usage:
//   ./fhe_microbench [--json] [--output FILE] [--iterations N]

#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <getopt.h>

#include "binfhecontext.h"
#include "math/hal/basicint.h"

#ifdef WITH_LUX_EXTENSIONS
#include "radix/radix.h"
#include "radix/shortint.h"
#endif

#include "../perf_thresholds.h"

using namespace lux::fhe;
using namespace std::chrono;

// ============================================================================
// Configuration
// ============================================================================

struct Config {
    int iterations = 100;
    int warmup = 10;
    bool json_output = false;
    std::string output_file;
    bool verbose = true;
    std::vector<int> batch_sizes = {1, 8, 32, 128};
};

// ============================================================================
// Benchmark Results Storage
// ============================================================================

struct MicrobenchResult {
    std::string category;
    std::string name;
    int batch_size;
    double latency_ms;
    double throughput_ops;
    double std_dev;
    int iterations;
    bool passed;
    double threshold;

    std::string to_json() const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(6);
        ss << "    {\n";
        ss << "      \"category\": \"" << category << "\",\n";
        ss << "      \"name\": \"" << name << "\",\n";
        ss << "      \"batch_size\": " << batch_size << ",\n";
        ss << "      \"latency_ms\": " << latency_ms << ",\n";
        ss << "      \"throughput_ops\": " << throughput_ops << ",\n";
        ss << "      \"std_dev\": " << std_dev << ",\n";
        ss << "      \"iterations\": " << iterations << ",\n";
        ss << "      \"passed\": " << (passed ? "true" : "false") << ",\n";
        ss << "      \"threshold\": " << threshold << "\n";
        ss << "    }";
        return ss.str();
    }
};

std::vector<MicrobenchResult> g_results;

// ============================================================================
// Timing Utilities
// ============================================================================

template<typename Func>
std::pair<double, double> time_operation(Func&& func, int iterations, int warmup) {
    // Warmup
    for (int i = 0; i < warmup; i++) {
        func();
    }

    // Collect samples
    std::vector<double> samples;
    samples.reserve(iterations);

    for (int i = 0; i < iterations; i++) {
        auto start = high_resolution_clock::now();
        func();
        auto end = high_resolution_clock::now();

        double elapsed = duration<double, std::milli>(end - start).count();
        samples.push_back(elapsed);
    }

    // Calculate mean
    double sum = 0.0;
    for (double s : samples) sum += s;
    double mean = sum / iterations;

    // Calculate std dev
    double var_sum = 0.0;
    for (double s : samples) {
        var_sum += (s - mean) * (s - mean);
    }
    double std_dev = std::sqrt(var_sum / iterations);

    return {mean, std_dev};
}

void print_result(const std::string& name, double latency, double throughput,
                  double threshold, bool passed) {
    std::cout << std::left << std::setw(40) << name
              << std::right << std::fixed << std::setprecision(3)
              << std::setw(10) << latency << " ms";
    if (throughput > 0) {
        std::cout << std::setw(14) << std::setprecision(1) << throughput << " ops/s";
    }
    std::cout << "  [" << (passed ? "PASS" : "FAIL") << "]" << std::endl;
}

// ============================================================================
// NTT Microbenchmarks
// ============================================================================

void benchmark_ntt(const Config& config) {
    std::cout << "\n=== NTT-4096 Benchmarks ===\n\n";

    uint32_t n = 4096;
    uint32_t m = n << 1;

    NativeInteger modulusQ(LastPrime<NativeInteger>(MAX_MODULUS_SIZE, m));
    NativeInteger rootOfUnity = RootOfUnity(m, modulusQ);

    DiscreteUniformGeneratorImpl<NativeVector> dug;

    ChineseRemainderTransformFTT<NativeVector> crtFTT;
    crtFTT.PreCompute(rootOfUnity, m, modulusQ);

    for (int batch_size : config.batch_sizes) {
        // Create batch of polynomials
        std::vector<NativeVector> polys(batch_size);
        std::vector<NativeVector> results(batch_size, NativeVector(n));

        for (int i = 0; i < batch_size; i++) {
            polys[i] = dug.GenerateVector(n, modulusQ);
        }

        // Forward NTT
        auto [fwd_time, fwd_std] = time_operation([&]() {
            for (int i = 0; i < batch_size; i++) {
                crtFTT.ForwardTransformToBitReverse(polys[i], rootOfUnity, m, &results[i]);
            }
        }, config.iterations, config.warmup);

        // Determine threshold based on batch size
        double threshold = 0.0;
        if (batch_size == 1) threshold = perf::thresholds::NTT_4096_BATCH_1.limit;
        else if (batch_size == 8) threshold = perf::thresholds::NTT_4096_BATCH_8.limit;
        else if (batch_size == 32) threshold = perf::thresholds::NTT_4096_BATCH_32.limit;
        else if (batch_size == 128) threshold = perf::thresholds::NTT_4096_BATCH_128.limit;
        else threshold = batch_size * 0.15; // Linear estimate

        bool passed = (fwd_time <= threshold * 1.15); // 15% tolerance
        double throughput = (batch_size * 1000.0) / fwd_time;

        std::stringstream ss;
        ss << "NTT-4096 Forward (batch=" << batch_size << ")";
        print_result(ss.str(), fwd_time, throughput, threshold, passed);

        g_results.push_back({
            "NTT", "Forward", batch_size,
            fwd_time, throughput, fwd_std, config.iterations,
            passed, threshold
        });

        // Inverse NTT
        auto [inv_time, inv_std] = time_operation([&]() {
            for (int i = 0; i < batch_size; i++) {
                crtFTT.InverseTransformFromBitReverse(results[i], rootOfUnity, m, &polys[i]);
            }
        }, config.iterations, config.warmup);

        passed = (inv_time <= threshold * 1.25); // INTT slightly slower
        throughput = (batch_size * 1000.0) / inv_time;

        ss.str("");
        ss << "NTT-4096 Inverse (batch=" << batch_size << ")";
        print_result(ss.str(), inv_time, throughput, threshold * 1.1, passed);

        g_results.push_back({
            "NTT", "Inverse", batch_size,
            inv_time, throughput, inv_std, config.iterations,
            passed, threshold * 1.1
        });

        // In-place variants
        auto [inplace_time, inplace_std] = time_operation([&]() {
            for (int i = 0; i < batch_size; i++) {
                crtFTT.ForwardTransformToBitReverseInPlace(rootOfUnity, m, &polys[i]);
            }
        }, config.iterations, config.warmup);

        passed = (inplace_time <= threshold);
        throughput = (batch_size * 1000.0) / inplace_time;

        ss.str("");
        ss << "NTT-4096 In-place (batch=" << batch_size << ")";
        print_result(ss.str(), inplace_time, throughput, threshold, passed);

        g_results.push_back({
            "NTT", "InPlace", batch_size,
            inplace_time, throughput, inplace_std, config.iterations,
            passed, threshold
        });
    }

    // Batch efficiency calculation
    if (g_results.size() >= 2) {
        double single_time = g_results[0].latency_ms; // batch=1
        for (int batch_size : {8, 32, 128}) {
            for (const auto& r : g_results) {
                if (r.batch_size == batch_size && r.name == "Forward") {
                    double ideal_time = single_time * batch_size;
                    double efficiency = (ideal_time / r.latency_ms) * 100.0;
                    std::cout << "  Batch=" << batch_size << " efficiency: "
                              << std::fixed << std::setprecision(1) << efficiency << "%\n";
                    break;
                }
            }
        }
    }
}

// ============================================================================
// Barrett Multiplication Benchmarks
// ============================================================================

void benchmark_barrett(const Config& config) {
    std::cout << "\n=== Barrett Multiplication Benchmarks ===\n\n";

    const size_t N = 100000;
    NativeInteger q = LastPrime<NativeInteger>(60, 8192);

    std::vector<NativeInteger> a(N), b(N), c(N);
    std::random_device rd;
    std::mt19937_64 gen(rd());

    uint64_t q_val = q.template ConvertToInt<uint64_t>();
    for (size_t i = 0; i < N; i++) {
        a[i] = NativeInteger(gen() % q_val);
        b[i] = NativeInteger(gen() % q_val);
    }

    // Batch sizes for Barrett
    for (int batch_size : {1000, 10000, 100000}) {
        if (static_cast<size_t>(batch_size) > N) break;

        auto [time_ms, std_dev] = time_operation([&]() {
            for (int i = 0; i < batch_size; i++) {
                c[i] = a[i].ModMul(b[i], q);
            }
        }, config.iterations, config.warmup);

        double throughput = (batch_size * 1000.0) / time_ms;
        double threshold = batch_size * 0.0001; // ~0.1us per op expected
        bool passed = (time_ms <= threshold);

        std::stringstream ss;
        ss << "BarrettMul (n=" << batch_size << ")";
        print_result(ss.str(), time_ms, throughput, threshold, passed);

        g_results.push_back({
            "Barrett", "ModMul", batch_size,
            time_ms, throughput, std_dev, config.iterations,
            passed, threshold
        });
    }
}

// ============================================================================
// Bootstrapping / External Product Benchmarks
// ============================================================================

void benchmark_bootstrap(const Config& config) {
    std::cout << "\n=== Bootstrapping Benchmarks ===\n\n";

    // Initialize context
    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(TOY, GINX);  // Use TOY for faster benchmarks
    auto sk = cc.KeyGen();
    cc.BTKeyGen(sk);

    // Single bootstrap (gate evaluation)
    auto ct1 = cc.Encrypt(sk, 1);
    auto ct0 = cc.Encrypt(sk, 0);

    // Single PBS via gate
    auto [single_time, single_std] = time_operation([&]() {
        cc.EvalBinGate(AND, ct1, ct0);
    }, config.iterations / 10, config.warmup / 2);

    double threshold = perf::thresholds::PBS_SINGLE.limit;
    bool passed = (single_time <= threshold * 1.15);

    print_result("PBS Single (TOY)", single_time, 1000.0 / single_time, threshold, passed);

    g_results.push_back({
        "PBS", "Single", 1,
        single_time, 1000.0 / single_time, single_std, config.iterations / 10,
        passed, threshold
    });

    // Batch PBS
    for (int batch_size : {4, 8, 16}) {
        std::vector<LWECiphertext> cts1(batch_size), cts0(batch_size);
        for (int i = 0; i < batch_size; i++) {
            cts1[i] = cc.Encrypt(sk, i % 2);
            cts0[i] = cc.Encrypt(sk, (i + 1) % 2);
        }

        auto [batch_time, batch_std] = time_operation([&]() {
            for (int i = 0; i < batch_size; i++) {
                cc.EvalBinGate(AND, cts1[i], cts0[i]);
            }
        }, config.iterations / 10, config.warmup / 2);

        double batch_threshold = single_time * batch_size * 0.9; // Should be ~linear
        passed = (batch_time <= batch_threshold);
        double throughput = (batch_size * 1000.0) / batch_time;

        std::stringstream ss;
        ss << "PBS Batch (n=" << batch_size << ")";
        print_result(ss.str(), batch_time, throughput, batch_threshold, passed);

        g_results.push_back({
            "PBS", "Batch", batch_size,
            batch_time, throughput, batch_std, config.iterations / 10,
            passed, batch_threshold
        });
    }
}

// ============================================================================
// Radix Operations Benchmarks (if WITH_LUX_EXTENSIONS)
// ============================================================================

#ifdef WITH_LUX_EXTENSIONS
void benchmark_radix_operations(const Config& config) {
    std::cout << "\n=== Radix Integer Benchmarks ===\n\n";

    using namespace radix;

    // Initialize context
    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(TOY, GINX);
    auto sk = cc.KeyGen();
    cc.BTKeyGen(sk);

    // Test different bit widths
    std::vector<std::pair<std::string, RadixParams>> widths = {
        {"8-bit", params::EUINT8},
        {"64-bit", params::EUINT64},
    };

    for (const auto& [name, params] : widths) {
        auto val1 = RadixInt::Encrypt(cc, params, 42, sk);
        auto val2 = RadixInt::Encrypt(cc, params, 17, sk);

        ShortIntLUTs luts(params.limb_params);

        // Addition
        auto [add_time, add_std] = time_operation([&]() {
            Add(val1, val2, luts);
        }, config.iterations / 20, config.warmup / 5);

        double threshold = params.num_limbs * 10.0; // ~10ms per limb
        bool passed = (add_time <= threshold * 1.2);

        std::stringstream ss;
        ss << "RadixAdd (" << name << ")";
        print_result(ss.str(), add_time, 1000.0 / add_time, threshold, passed);

        g_results.push_back({
            "Radix", "Add-" + name, static_cast<int>(params.num_limbs),
            add_time, 1000.0 / add_time, add_std, config.iterations / 20,
            passed, threshold
        });

        // Comparison
        auto [cmp_time, cmp_std] = time_operation([&]() {
            Lt(val1, val2, luts);
        }, config.iterations / 20, config.warmup / 5);

        threshold = params.num_limbs * 15.0; // Comparisons slightly more expensive
        passed = (cmp_time <= threshold * 1.2);

        ss.str("");
        ss << "RadixLt (" << name << ")";
        print_result(ss.str(), cmp_time, 1000.0 / cmp_time, threshold, passed);

        g_results.push_back({
            "Radix", "Lt-" + name, static_cast<int>(params.num_limbs),
            cmp_time, 1000.0 / cmp_time, cmp_std, config.iterations / 20,
            passed, threshold
        });
    }
}

void benchmark_carry_propagation(const Config& config) {
    std::cout << "\n=== Carry Propagation (Normalize) Benchmarks ===\n\n";

    using namespace radix;

    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(TOY, GINX);
    auto sk = cc.KeyGen();
    cc.BTKeyGen(sk);

    // Test different depths (number of limbs)
    std::vector<std::pair<int, RadixParams>> depths = {
        {4, params::EUINT8},    // depth=4 (8-bit)
        {8, {{2, 2}, 8}},       // depth=8 (16-bit custom)
        {32, params::EUINT64},  // depth=32 (64-bit)
    };

    for (const auto& [depth, params] : depths) {
        auto val = RadixInt::Encrypt(cc, params, 255, sk);

        ShortIntLUTs luts(params.limb_params);

        // Perform additions to accumulate carries
        for (int i = 0; i < 3; i++) {
            val.AddScalarInPlace(100, luts);
        }

        // Benchmark PropagateCarries (normalize)
        auto [time_ms, std_dev] = time_operation([&]() {
            val.PropagateCarries(luts);
        }, config.iterations / 20, config.warmup / 5);

        double threshold = 0.0;
        if (depth == 4) threshold = perf::thresholds::NORMALIZE_DEPTH_4.limit;
        else if (depth == 8) threshold = perf::thresholds::NORMALIZE_DEPTH_8.limit;
        else threshold = depth * 15.0;

        bool passed = (time_ms <= threshold * 1.2);

        std::stringstream ss;
        ss << "Normalize (depth=" << depth << ")";
        print_result(ss.str(), time_ms, 1000.0 / time_ms, threshold, passed);

        g_results.push_back({
            "Normalize", "PropagateCarries", depth,
            time_ms, 1000.0 / time_ms, std_dev, config.iterations / 20,
            passed, threshold
        });
    }
}

void benchmark_shift_operations(const Config& config) {
    std::cout << "\n=== Shift Operation Benchmarks ===\n\n";

    using namespace radix;

    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(TOY, GINX);
    auto sk = cc.KeyGen();
    cc.BTKeyGen(sk);

    RadixParams params = params::EUINT64;
    auto val = RadixInt::Encrypt(cc, params, 0x12345678, sk);
    ShortIntLUTs luts(params.limb_params);

    // Byte-aligned shift (cheap: just moves limbs)
    std::vector<int> byte_shifts = {8, 16, 32};
    for (int shift : byte_shifts) {
        auto [time_ms, std_dev] = time_operation([&]() {
            RadixInt copy = val;
            copy.ShlInPlace(shift, luts);
        }, config.iterations / 10, config.warmup / 2);

        double threshold = perf::thresholds::SHIFT_BYTE_ALIGNED.limit;
        bool passed = (time_ms <= threshold * 1.2);

        std::stringstream ss;
        ss << "Shl ByteAligned (shift=" << shift << ")";
        print_result(ss.str(), time_ms, 1000.0 / time_ms, threshold, passed);

        g_results.push_back({
            "Shift", "ByteAligned", shift,
            time_ms, 1000.0 / time_ms, std_dev, config.iterations / 10,
            passed, threshold
        });
    }

    // General shift (more expensive: requires bit manipulation)
    std::vector<int> general_shifts = {1, 3, 7, 13};
    for (int shift : general_shifts) {
        auto [time_ms, std_dev] = time_operation([&]() {
            RadixInt copy = val;
            copy.ShlInPlace(shift, luts);
        }, config.iterations / 10, config.warmup / 2);

        double threshold = perf::thresholds::SHIFT_GENERAL.limit;
        bool passed = (time_ms <= threshold * 1.2);

        std::stringstream ss;
        ss << "Shl General (shift=" << shift << ")";
        print_result(ss.str(), time_ms, 1000.0 / time_ms, threshold, passed);

        g_results.push_back({
            "Shift", "General", shift,
            time_ms, 1000.0 / time_ms, std_dev, config.iterations / 10,
            passed, threshold
        });
    }
}
#endif // WITH_LUX_EXTENSIONS

// ============================================================================
// Output Functions
// ============================================================================

void output_json(const Config& config) {
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"timestamp\": \"" << duration_cast<seconds>(
        system_clock::now().time_since_epoch()).count() << "\",\n";
    ss << "  \"iterations\": " << config.iterations << ",\n";
    ss << "  \"warmup\": " << config.warmup << ",\n";
    ss << "  \"results\": [\n";

    for (size_t i = 0; i < g_results.size(); i++) {
        ss << g_results[i].to_json();
        if (i < g_results.size() - 1) ss << ",";
        ss << "\n";
    }

    ss << "  ],\n";

    // Summary stats
    int total = g_results.size();
    int passed = 0;
    for (const auto& r : g_results) {
        if (r.passed) passed++;
    }

    ss << "  \"summary\": {\n";
    ss << "    \"total\": " << total << ",\n";
    ss << "    \"passed\": " << passed << ",\n";
    ss << "    \"failed\": " << (total - passed) << "\n";
    ss << "  }\n";
    ss << "}\n";

    if (config.output_file.empty()) {
        std::cout << "\n=== JSON Output ===\n" << ss.str();
    } else {
        std::ofstream out(config.output_file);
        out << ss.str();
        std::cout << "\nJSON written to: " << config.output_file << "\n";
    }
}

void print_summary() {
    std::cout << "\n=== Summary ===\n";

    int total = g_results.size();
    int passed = 0;
    for (const auto& r : g_results) {
        if (r.passed) passed++;
    }

    std::cout << "Total benchmarks: " << total << "\n";
    std::cout << "Passed: " << passed << "\n";
    std::cout << "Failed: " << (total - passed) << "\n";

    if (passed < total) {
        std::cout << "\nFailed benchmarks:\n";
        for (const auto& r : g_results) {
            if (!r.passed) {
                std::cout << "  - " << r.category << "/" << r.name
                          << " (batch=" << r.batch_size << "): "
                          << std::fixed << std::setprecision(3) << r.latency_ms
                          << " ms > " << r.threshold << " ms\n";
            }
        }
    }
}

// ============================================================================
// Main
// ============================================================================

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --json           Output results in JSON format\n";
    std::cout << "  --output FILE    Write JSON to FILE\n";
    std::cout << "  --iterations N   Number of benchmark iterations (default: 100)\n";
    std::cout << "  --warmup N       Number of warmup iterations (default: 10)\n";
    std::cout << "  --quiet          Suppress verbose output\n";
    std::cout << "  --help           Show this help message\n";
}

int main(int argc, char* argv[]) {
    Config config;

    static struct option long_options[] = {
        {"json", no_argument, nullptr, 'j'},
        {"output", required_argument, nullptr, 'o'},
        {"iterations", required_argument, nullptr, 'n'},
        {"warmup", required_argument, nullptr, 'w'},
        {"quiet", no_argument, nullptr, 'q'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "jo:n:w:qh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'j':
                config.json_output = true;
                break;
            case 'o':
                config.output_file = optarg;
                config.json_output = true;
                break;
            case 'n':
                config.iterations = std::stoi(optarg);
                break;
            case 'w':
                config.warmup = std::stoi(optarg);
                break;
            case 'q':
                config.verbose = false;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    std::cout << "========================================\n";
    std::cout << "  Lux FHE Microbenchmark Suite\n";
    std::cout << "========================================\n";
    std::cout << "\nConfiguration:\n";
    std::cout << "  Iterations: " << config.iterations << "\n";
    std::cout << "  Warmup: " << config.warmup << "\n";
    std::cout << "  JSON output: " << (config.json_output ? "yes" : "no") << "\n";

    // Run benchmarks
    benchmark_ntt(config);
    benchmark_barrett(config);
    benchmark_bootstrap(config);

#ifdef WITH_LUX_EXTENSIONS
    benchmark_radix_operations(config);
    benchmark_carry_propagation(config);
    benchmark_shift_operations(config);
#else
    std::cout << "\n[Radix benchmarks skipped - build with WITH_LUX_EXTENSIONS=ON]\n";
#endif

    // Output
    if (config.json_output) {
        output_json(config);
    }

    print_summary();

    // Return non-zero if any benchmark failed
    int failures = 0;
    for (const auto& r : g_results) {
        if (!r.passed) failures++;
    }

    return (failures > 0) ? 1 : 0;
}
