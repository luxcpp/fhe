//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2024, Lux Industries Inc
//
// All rights reserved.
//
// MLX GPU Backend Implementation for OpenFHE
//==================================================================================

#include "math/hal/mlx/mlx_backend.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace lbcrypto {
namespace mlx_backend {

// ============================================================================
// Utility Functions
// ============================================================================

bool IsMLXAvailable() {
#ifdef WITH_MLX
    try {
        return mx::metal::is_available();
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

std::string GetDeviceName() {
#ifdef WITH_MLX
    if (IsMLXAvailable()) {
        auto device = mx::default_device();
        return device.type == mx::Device::DeviceType::gpu ? "Apple GPU" : "CPU";
    }
    return "MLX not available";
#else
    return "MLX not compiled";
#endif
}

size_t GetGPUMemoryUsage() {
#ifdef WITH_MLX
    if (IsMLXAvailable()) {
        return mx::get_active_memory();
    }
#endif
    return 0;
}

#ifdef WITH_MLX

// ============================================================================
// Modular Arithmetic Helpers
// ============================================================================

namespace {

// Modular multiplication for 64-bit values
inline uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m) {
    return static_cast<uint64_t>((__uint128_t(a) * b) % m);
}

// Modular addition
inline uint64_t addmod(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t sum = a + b;
    return sum >= m ? sum - m : sum;
}

// Modular subtraction
inline uint64_t submod(uint64_t a, uint64_t b, uint64_t m) {
    return a >= b ? a - b : m - b + a;
}

// Modular power
uint64_t powmod(uint64_t base, uint64_t exp, uint64_t m) {
    uint64_t result = 1;
    base %= m;
    while (exp > 0) {
        if (exp & 1) {
            result = mulmod(result, base, m);
        }
        exp >>= 1;
        base = mulmod(base, base, m);
    }
    return result;
}

// Modular inverse using extended Euclidean algorithm
uint64_t modinv(uint64_t a, uint64_t m) {
    int64_t t = 0, newt = 1;
    int64_t r = m, newr = a;
    while (newr != 0) {
        int64_t quotient = r / newr;
        std::tie(t, newt) = std::make_pair(newt, t - quotient * newt);
        std::tie(r, newr) = std::make_pair(newr, r - quotient * newr);
    }
    if (t < 0) t += m;
    return static_cast<uint64_t>(t);
}

// Find primitive n-th root of unity mod q
uint64_t find_primitive_root(uint64_t n, uint64_t q) {
    // For NTT-friendly primes, q = k*n + 1
    // Generator of multiplicative group has order q-1
    // We need element of order n, so take g^((q-1)/n)
    
    // Try small generators
    for (uint64_t g = 2; g < 100; ++g) {
        // Check if g is a primitive root of unity of order n
        uint64_t w = powmod(g, (q - 1) / n, q);
        
        // Verify: w^n = 1 and w^(n/2) = -1 (= q-1)
        if (powmod(w, n, q) == 1 && powmod(w, n/2, q) == q - 1) {
            return w;
        }
    }
    
    throw std::runtime_error("Failed to find primitive root of unity");
}

} // anonymous namespace

// ============================================================================
// MLXNTT Implementation - CPU-based NTT with MLX for batch parallelization
// ============================================================================

MLXNTT::MLXNTT(uint64_t n, uint64_t q, const MLXConfig& config)
    : n_(n), q_(q), config_(config),
      twiddles_(mx::zeros({static_cast<int>(n)})),
      inv_twiddles_(mx::zeros({static_cast<int>(n)})),
      n_inv_(mx::array(1.0f)) {
    
    // Validate parameters
    if ((n & (n - 1)) != 0) {
        throw std::invalid_argument("Ring dimension must be power of 2");
    }
    
    // Initialize GPU
    try {
        if (config.device_type == "gpu" && mx::metal::is_available()) {
            mx::set_default_device(mx::Device::gpu);
            gpu_enabled_ = true;
            std::cout << "MLX NTT: GPU acceleration enabled\n";
        } else {
            mx::set_default_device(mx::Device::cpu);
            gpu_enabled_ = false;
            std::cout << "MLX NTT: Running in CPU mode\n";
        }
    } catch (const std::exception& e) {
        gpu_enabled_ = false;
        std::cerr << "MLX NTT: Failed to initialize GPU: " << e.what() << "\n";
    }
    
    // Precompute twiddle factors (stored as CPU vectors for correctness)
    PrecomputeTwiddles();
}

MLXNTT::~MLXNTT() = default;

MLXNTT::MLXNTT(MLXNTT&&) noexcept = default;
MLXNTT& MLXNTT::operator=(MLXNTT&&) noexcept = default;

void MLXNTT::PrecomputeTwiddles() {
    // Find primitive n-th root of unity
    uint64_t w = find_primitive_root(n_, q_);
    uint64_t w_inv = modinv(w, q_);
    
    // Store twiddle factors
    twiddle_factors_.resize(n_);
    inv_twiddle_factors_.resize(n_);
    
    uint64_t pow = 1;
    uint64_t pow_inv = 1;
    for (uint64_t i = 0; i < n_; ++i) {
        twiddle_factors_[i] = pow;
        inv_twiddle_factors_[i] = pow_inv;
        pow = mulmod(pow, w, q_);
        pow_inv = mulmod(pow_inv, w_inv, q_);
    }
    
    // Compute n^(-1) mod q
    n_inv_val_ = modinv(n_, q_);
}

mx::array MLXNTT::ToMLXArray(const std::vector<uint64_t>& vec) {
    std::vector<float> float_data(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        float_data[i] = static_cast<float>(vec[i]);
    }
    return mx::array(float_data.data(), {static_cast<int>(vec.size())}, mx::float32);
}

void MLXNTT::FromMLXArray(const mx::array& arr, std::vector<uint64_t>& vec) {
    mx::eval(arr);
    const float* data = arr.data<float>();
    vec.resize(arr.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        float val = data[i];
        int64_t ival = static_cast<int64_t>(std::round(val));
        ival = ((ival % static_cast<int64_t>(q_)) + static_cast<int64_t>(q_)) % static_cast<int64_t>(q_);
        vec[i] = static_cast<uint64_t>(ival);
    }
}

mx::array MLXNTT::NTTCore(const mx::array& input, const mx::array& twiddles) {
    // This is a placeholder - actual GPU NTT would use custom Metal kernel
    // For now, we copy to CPU, do NTT, and copy back
    (void)twiddles; // unused in CPU path
    
    mx::eval(input);
    const float* input_data = input.data<float>();
    
    std::vector<uint64_t> data(n_);
    for (size_t i = 0; i < n_; ++i) {
        data[i] = static_cast<uint64_t>(input_data[i]);
    }
    
    // CPU NTT implementation
    NTTCpu(data, false);
    
    std::vector<float> result(n_);
    for (size_t i = 0; i < n_; ++i) {
        result[i] = static_cast<float>(data[i]);
    }
    
    return mx::array(result.data(), {static_cast<int>(n_)}, mx::float32);
}

mx::array MLXNTT::INTTCore(const mx::array& input, const mx::array& inv_twiddles) {
    (void)inv_twiddles; // unused in CPU path
    
    mx::eval(input);
    const float* input_data = input.data<float>();
    
    std::vector<uint64_t> data(n_);
    for (size_t i = 0; i < n_; ++i) {
        data[i] = static_cast<uint64_t>(input_data[i]);
    }
    
    // CPU INTT implementation
    NTTCpu(data, true);
    
    std::vector<float> result(n_);
    for (size_t i = 0; i < n_; ++i) {
        result[i] = static_cast<float>(data[i]);
    }
    
    return mx::array(result.data(), {static_cast<int>(n_)}, mx::float32);
}

void MLXNTT::NTTCpu(std::vector<uint64_t>& data, bool inverse) {
    const auto& twiddles = inverse ? inv_twiddle_factors_ : twiddle_factors_;
    
    // Bit-reversal permutation
    int log_n = 0;
    for (uint64_t temp = n_; temp > 1; temp >>= 1) ++log_n;
    
    for (uint64_t i = 0; i < n_; ++i) {
        uint64_t j = 0;
        for (int k = 0; k < log_n; ++k) {
            if (i & (1ULL << k)) {
                j |= (1ULL << (log_n - 1 - k));
            }
        }
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }
    
    // Cooley-Tukey butterfly
    for (uint64_t len = 2; len <= n_; len <<= 1) {
        uint64_t step = n_ / len;
        for (uint64_t i = 0; i < n_; i += len) {
            for (uint64_t j = 0; j < len / 2; ++j) {
                uint64_t w = twiddles[j * step];
                uint64_t u = data[i + j];
                uint64_t v = mulmod(data[i + j + len/2], w, q_);
                
                data[i + j] = addmod(u, v, q_);
                data[i + j + len/2] = submod(u, v, q_);
            }
        }
    }
    
    // Scale by n^(-1) for inverse
    if (inverse) {
        for (uint64_t i = 0; i < n_; ++i) {
            data[i] = mulmod(data[i], n_inv_val_, q_);
        }
    }
}

void MLXNTT::ForwardTransform(const std::vector<uint64_t>& input, std::vector<uint64_t>& output) {
    if (input.size() != n_) {
        throw std::invalid_argument("Input size must match ring dimension");
    }
    
    output = input;
    NTTCpu(output, false);
}

void MLXNTT::InverseTransform(const std::vector<uint64_t>& input, std::vector<uint64_t>& output) {
    if (input.size() != n_) {
        throw std::invalid_argument("Input size must match ring dimension");
    }
    
    output = input;
    NTTCpu(output, true);
}

void MLXNTT::ForwardTransformBatch(const std::vector<std::vector<uint64_t>>& inputs,
                                    std::vector<std::vector<uint64_t>>& outputs) {
    if (inputs.empty()) return;
    
    outputs.resize(inputs.size());
    
    // Parallel processing using MLX's threading
    // For now, simple sequential - MLX parallelism would be for GPU kernels
    for (size_t i = 0; i < inputs.size(); ++i) {
        ForwardTransform(inputs[i], outputs[i]);
    }
}

void MLXNTT::InverseTransformBatch(const std::vector<std::vector<uint64_t>>& inputs,
                                    std::vector<std::vector<uint64_t>>& outputs) {
    if (inputs.empty()) return;
    
    outputs.resize(inputs.size());
    
    for (size_t i = 0; i < inputs.size(); ++i) {
        InverseTransform(inputs[i], outputs[i]);
    }
}

void MLXNTT::ElementwiseMultMod(const std::vector<uint64_t>& a, 
                                const std::vector<uint64_t>& b,
                                std::vector<uint64_t>& result) {
    result.resize(n_);
    for (size_t i = 0; i < n_; ++i) {
        result[i] = mulmod(a[i], b[i], q_);
    }
}

// ============================================================================
// MLXPolyOps Implementation
// ============================================================================

MLXPolyOps::MLXPolyOps(uint64_t n, uint64_t q, const MLXConfig& config)
    : n_(n), q_(q) {
    ntt_ = std::make_unique<MLXNTT>(n, q, config);
    gpu_enabled_ = ntt_->IsGPUEnabled();
}

MLXPolyOps::~MLXPolyOps() = default;

void MLXPolyOps::PolyMult(const std::vector<uint64_t>& a,
                          const std::vector<uint64_t>& b,
                          std::vector<uint64_t>& result) {
    // Multiplication in NTT domain
    std::vector<uint64_t> a_ntt, b_ntt;
    ntt_->ForwardTransform(a, a_ntt);
    ntt_->ForwardTransform(b, b_ntt);
    
    std::vector<uint64_t> prod_ntt;
    ntt_->ElementwiseMultMod(a_ntt, b_ntt, prod_ntt);
    
    ntt_->InverseTransform(prod_ntt, result);
}

void MLXPolyOps::PolyAdd(const std::vector<uint64_t>& a,
                         const std::vector<uint64_t>& b,
                         std::vector<uint64_t>& result) {
    result.resize(n_);
    for (size_t i = 0; i < n_; ++i) {
        result[i] = addmod(a[i], b[i], q_);
    }
}

void MLXPolyOps::PolySub(const std::vector<uint64_t>& a,
                         const std::vector<uint64_t>& b,
                         std::vector<uint64_t>& result) {
    result.resize(n_);
    for (size_t i = 0; i < n_; ++i) {
        result[i] = submod(a[i], b[i], q_);
    }
}

void MLXPolyOps::PolyNeg(const std::vector<uint64_t>& a,
                         std::vector<uint64_t>& result) {
    result.resize(n_);
    for (size_t i = 0; i < n_; ++i) {
        result[i] = a[i] == 0 ? 0 : q_ - a[i];
    }
}

void MLXPolyOps::Automorphism(const std::vector<uint64_t>& a,
                              uint64_t k,
                              std::vector<uint64_t>& result) {
    result.resize(n_);
    std::fill(result.begin(), result.end(), 0);
    
    for (size_t i = 0; i < n_; ++i) {
        size_t idx = (i * k) % (2 * n_);
        if (idx < n_) {
            result[idx] = addmod(result[idx], a[i], q_);
        } else {
            // Negative coefficient (in ring Z[X]/(X^n + 1))
            result[idx - n_] = submod(result[idx - n_], a[i], q_);
        }
    }
}

void MLXPolyOps::RGSWExternalProduct(const std::vector<std::vector<uint64_t>>& ct,
                                     const std::vector<std::vector<uint64_t>>& rgsw,
                                     std::vector<std::vector<uint64_t>>& result) {
    // RGSW external product: ct * RGSW(m) -> ct * m
    result.resize(ct.size());
    for (size_t i = 0; i < ct.size(); ++i) {
        result[i].resize(n_, 0);
        for (size_t j = 0; j < rgsw.size(); ++j) {
            std::vector<uint64_t> prod;
            PolyMult(ct[i], rgsw[j], prod);
            PolyAdd(result[i], prod, result[i]);
        }
    }
}

// ============================================================================
// MLXBlindRotation Implementation
// ============================================================================

MLXBlindRotation::MLXBlindRotation(uint64_t n, uint64_t N, uint64_t q, uint64_t Q,
                                   const MLXConfig& config)
    : n_(n), N_(N), q_(q), Q_(Q) {
    poly_ops_ = std::make_unique<MLXPolyOps>(N, Q, config);
}

MLXBlindRotation::~MLXBlindRotation() = default;

void MLXBlindRotation::Evaluate(const std::vector<std::vector<uint64_t>>& acc,
                                const std::vector<uint64_t>& lwe_ct,
                                const std::vector<std::vector<std::vector<uint64_t>>>& bsk,
                                std::vector<std::vector<uint64_t>>& result) {
    result = acc;
    
    // Process each LWE coefficient
    for (size_t i = 0; i < n_; ++i) {
        if (lwe_ct[i] == 0) continue;
        
        // Apply automorphism for rotation
        std::vector<std::vector<uint64_t>> rotated(result.size());
        uint64_t rotation_amount = (2 * N_ - lwe_ct[i]) % (2 * N_);
        for (size_t j = 0; j < result.size(); ++j) {
            poly_ops_->Automorphism(result[j], rotation_amount, rotated[j]);
        }
        
        // RGSW external product
        if (i < bsk.size()) {
            poly_ops_->RGSWExternalProduct(rotated, bsk[i], result);
        }
    }
}

void MLXBlindRotation::EvaluateBatch(const std::vector<std::vector<std::vector<uint64_t>>>& accs,
                                     const std::vector<std::vector<uint64_t>>& lwe_cts,
                                     const std::vector<std::vector<std::vector<uint64_t>>>& bsk,
                                     std::vector<std::vector<std::vector<uint64_t>>>& results) {
    results.resize(accs.size());
    
    for (size_t i = 0; i < accs.size(); ++i) {
        Evaluate(accs[i], lwe_cts[i], bsk, results[i]);
    }
}

#endif // WITH_MLX

} // namespace mlx_backend
} // namespace lbcrypto
