// =============================================================================
// GPU-Accelerated Key Switching for Lux FHE
// =============================================================================
//
// Key switching converts an RLWE ciphertext under one key to an LWE ciphertext
// under a different key, completing the bootstrap process.
//
// Algorithm:
// 1. Extract constant coefficient from RLWE accumulator
// 2. Decompose the result using base-B digits
// 3. Multiply decomposed digits by key switching key components
// 4. Sum to get final LWE ciphertext
//
// The key switching key (KSK) structure:
// - For each coefficient i in [0, N-1]
// - For each decomposition level l in [0, L_ks-1]
// - KSK[i][l] is an LWE encryption of floor(s_i * B^l)

#ifndef LUX_FHE_MATH_HAL_MLX_KEY_SWITCH_GPU_H
#define LUX_FHE_MATH_HAL_MLX_KEY_SWITCH_GPU_H

#include <cstdint>
#include <vector>
#include <memory>

#ifdef WITH_MLX
#include <mlx/mlx.h>
namespace mx = mlx::core;
#endif

namespace lux {
namespace gpu {

#ifdef WITH_MLX

// =============================================================================
// Key Switching Engine
// =============================================================================

class KeySwitch {
public:
    struct Config {
        uint32_t N;           // Ring dimension (source)
        uint32_t n;           // LWE dimension (target)
        uint32_t L_ks;        // Key switching decomposition levels
        uint32_t baseLog_ks;  // log2(key switching base)
        uint64_t Q;           // Modulus
        uint64_t q_lwe;       // LWE modulus (may be smaller)
    };
    
    KeySwitch(const Config& config);
    ~KeySwitch() = default;
    
    // Main key switching operation
    // rlwe: [B, 2, N] - RLWE ciphertexts from blind rotation
    // ksk: [N, L_ks, n+1] - key switching key
    // Returns: [B, n+1] - LWE ciphertexts
    mx::array keySwitch(const mx::array& rlwe, const mx::array& ksk);
    
    // Extract constant coefficient from RLWE
    // rlwe: [B, 2, N] - RLWE ciphertexts
    // Returns: [B, 2] - (a[0], b) for each ciphertext
    mx::array extractConstant(const mx::array& rlwe);
    
    // Modulus switching (scale from Q to q_lwe)
    // ct: [B, n+1] - ciphertexts mod Q
    // Returns: [B, n+1] - ciphertexts mod q_lwe
    mx::array modulusSwitch(const mx::array& ct);
    
private:
    Config config_;
    uint64_t base_ks_;
    uint64_t mask_ks_;
    
    static inline uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m) {
        __uint128_t product = static_cast<__uint128_t>(a) * b;
        return static_cast<uint64_t>(product % m);
    }
    
    static inline uint64_t addmod(uint64_t a, uint64_t b, uint64_t m) {
        uint64_t sum = a + b;
        return (sum >= m) ? sum - m : sum;
    }
    
    static inline uint64_t submod(uint64_t a, uint64_t b, uint64_t m) {
        return (a >= b) ? a - b : a + m - b;
    }
    
    // Round division for modulus switching
    static inline uint64_t round_div(uint64_t x, uint64_t y) {
        return (x + y / 2) / y;
    }
};

// =============================================================================
// Implementation
// =============================================================================

inline KeySwitch::KeySwitch(const Config& config)
    : config_(config) {
    base_ks_ = 1ULL << config_.baseLog_ks;
    mask_ks_ = base_ks_ - 1;
}

inline mx::array KeySwitch::keySwitch(const mx::array& rlwe,
                                          const mx::array& ksk) {
    // rlwe: [B, 2, N] - RLWE ciphertexts
    // ksk: [N, L_ks, n+1] - key switching key
    // Output: [B, n+1] - LWE ciphertexts
    
    auto shape = rlwe.shape();
    int B = shape[0];
    int N = shape[2];
    int n = config_.n;
    uint32_t L_ks = config_.L_ks;
    uint64_t Q = config_.Q;
    
    mx::eval(rlwe);
    mx::eval(ksk);
    
    auto rlwePtr = rlwe.data<int64_t>();
    auto kskPtr = ksk.data<int64_t>();
    
    std::vector<int64_t> resultData(B * (n + 1), 0);
    
    for (int b = 0; b < B; ++b) {
        const int64_t* c0 = rlwePtr + b * 2 * N;      // First polynomial
        const int64_t* c1 = rlwePtr + b * 2 * N + N;  // Second polynomial
        int64_t* lwe = resultData.data() + b * (n + 1);
        
        // Initialize result: b = c1[0] (constant term of second polynomial)
        lwe[n] = c1[0];
        
        // Key switching on first polynomial c0
        // For each coefficient of c0, decompose and multiply by KSK
        for (int i = 0; i < N; ++i) {
            uint64_t coeff = static_cast<uint64_t>(c0[i]) % Q;
            
            // Decompose coefficient into L_ks digits
            for (uint32_t l = 0; l < L_ks; ++l) {
                uint64_t digit = (coeff >> (l * config_.baseLog_ks)) & mask_ks_;
                
                if (digit == 0) continue;
                
                // Get KSK[i][l]: an LWE ciphertext (n+1 elements)
                const int64_t* kskEntry = kskPtr + i * L_ks * (n + 1) + l * (n + 1);
                
                // Accumulate: result += digit * KSK[i][l]
                for (int j = 0; j <= n; ++j) {
                    uint64_t prod = mulmod(digit, 
                                           static_cast<uint64_t>(kskEntry[j]) % Q, Q);
                    lwe[j] = static_cast<int64_t>(
                        addmod(static_cast<uint64_t>(lwe[j]) % Q, prod, Q));
                }
            }
        }
    }
    
    return mx::array(resultData.data(), {B, n + 1}, mx::int64);
}

inline mx::array KeySwitch::extractConstant(const mx::array& rlwe) {
    // rlwe: [B, 2, N]
    // Output: [B, 2] containing (c0[0], c1[0])
    
    auto shape = rlwe.shape();
    int B = shape[0];
    int N = shape[2];
    
    mx::eval(rlwe);
    auto rlwePtr = rlwe.data<int64_t>();
    
    std::vector<int64_t> resultData(B * 2);
    
    for (int b = 0; b < B; ++b) {
        resultData[b * 2] = rlwePtr[b * 2 * N];      // c0[0]
        resultData[b * 2 + 1] = rlwePtr[b * 2 * N + N];  // c1[0]
    }
    
    return mx::array(resultData.data(), {B, 2}, mx::int64);
}

inline mx::array KeySwitch::modulusSwitch(const mx::array& ct) {
    // ct: [B, n+1] mod Q
    // Output: [B, n+1] mod q_lwe
    
    auto shape = ct.shape();
    int B = shape[0];
    int len = shape[1];
    uint64_t Q = config_.Q;
    uint64_t q = config_.q_lwe;
    
    mx::eval(ct);
    auto ctPtr = ct.data<int64_t>();
    
    std::vector<int64_t> resultData(B * len);
    
    for (int i = 0; i < B * len; ++i) {
        uint64_t val = static_cast<uint64_t>(ctPtr[i]) % Q;
        // Scale: round(val * q / Q)
        resultData[i] = static_cast<int64_t>(round_div(val * q, Q));
    }
    
    return mx::array(resultData.data(), shape, mx::int64);
}

#endif // WITH_MLX

} // namespace gpu
} // namespace lux::fhe

#endif // LUX_FHE_MATH_HAL_MLX_KEY_SWITCH_GPU_H
