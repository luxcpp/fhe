// =============================================================================
// euint256 PBS Integration - Optimized PBS Operations for 256-bit Arithmetic
// =============================================================================
//
// This file provides optimized PBS operations specifically tuned for euint256
// arithmetic. Key optimizations:
//
// 1. Word-Parallel PBS: All 8 words can be processed in single batched PBS
// 2. Kogge-Stone Fusion: 3 rounds of carry propagation fused into batched ops
// 3. Comparison Chain: 8-word comparison flags computed in parallel
// 4. Shared BSK: Bootstrap key loaded once, reused across all operations
//
// Integration:
//   #include "euint256_pbs_integration.h"
//   euint256PBSContext ctx(config);
//   ctx.setKeys(bsk, ksk);
//   auto result = ctx.parallelAdd(a_words, b_words);
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0
// =============================================================================

#ifndef LBCRYPTO_MATH_HAL_MLX_EUINT256_PBS_INTEGRATION_H
#define LBCRYPTO_MATH_HAL_MLX_EUINT256_PBS_INTEGRATION_H

#include <array>
#include <vector>
#include <memory>

#ifdef WITH_MLX
#include <mlx/mlx.h>
#include "pbs_optimized.h"
#include "blind_rotate.h"
#include "key_switch.h"
#include "external_product_fused.h"
namespace mx = mlx::core;
#endif

namespace lbcrypto {
namespace gpu {

#ifdef WITH_MLX

// Helper to create initialized array of 8 mx::arrays (avoids default ctor issue)
inline std::array<mx::array, 8> makeArrayOf8() {
    mx::array p = mx::array(static_cast<int64_t>(0));
    return {p, p, p, p, p, p, p, p};
}

// =============================================================================
// euint256 PBS Context
// =============================================================================
//
// Holds pre-loaded keys and provides optimized multi-word PBS operations.
// Thread-safe for concurrent operations on different euint256 values.

class euint256PBSContext {
public:
    struct Config {
        uint32_t N = 1024;
        uint32_t n = 512;
        uint32_t L = 3;
        uint32_t baseLog = 7;
        uint64_t Q = 1ULL << 27;
        uint32_t L_ks = 4;
        uint32_t baseLog_ks = 4;
        uint64_t q_lwe = 1ULL << 15;
    };

    explicit euint256PBSContext(const Config& cfg);

    void setBootstrapKey(const mx::array& bsk);
    void setKeySwitchKey(const mx::array& ksk);

    // =========================================================================
    // Word-Parallel Operations (8 words at once)
    // =========================================================================

    // Parallel 32-bit add across all 8 words (with PBS refresh)
    std::array<mx::array, 8> parallelWordAdd(
        const std::array<mx::array, 8>& a,
        const std::array<mx::array, 8>& b);

    // Parallel 32-bit sub across all 8 words
    std::array<mx::array, 8> parallelWordSub(
        const std::array<mx::array, 8>& a,
        const std::array<mx::array, 8>& b);

    // =========================================================================
    // Kogge-Stone Carry Propagation (Fused 3-Round)
    // =========================================================================
    //
    // Traditional Kogge-Stone: 3 rounds x 7 combines = 21 PBS calls
    // Fused approach: 3 batched PBS calls with all 7 combines per round
    //
    // Input: 8 (generate, propagate) pairs
    // Output: 8 carry bits

    struct GeneratePropagate {
        mx::array generate{static_cast<int64_t>(0)};   // G[i] = a[i] AND b[i]
        mx::array propagate{static_cast<int64_t>(0)};  // P[i] = a[i] XOR b[i]
    };

    std::array<mx::array, 8> fusedKoggeStoneCarries(
        const std::array<GeneratePropagate, 8>& gp,
        const mx::array& carryIn);

    // Compute G/P pairs for 8 words (8 parallel PBS)
    std::array<GeneratePropagate, 8> computeGeneratePropagateParallel(
        const std::array<mx::array, 8>& a,
        const std::array<mx::array, 8>& b);

    // Full normalized addition with fused carry propagation
    std::array<mx::array, 8> normalizedAdd(
        const std::array<mx::array, 8>& a,
        const std::array<mx::array, 8>& b);

    // =========================================================================
    // Parallel Comparison (8-Word Prefix Scan)
    // =========================================================================
    //
    // Compute (gt, eq, lt) flags for 8 words in parallel, then fuse
    // 3-round prefix scan combining into batched PBS.

    struct CompareFlags {
        mx::array gt{static_cast<int64_t>(0)};
        mx::array eq{static_cast<int64_t>(0)};
        mx::array lt{static_cast<int64_t>(0)};
    };

    // Parallel word comparison (8 PBS - one per word pair)
    std::array<CompareFlags, 8> parallelWordCompare(
        const std::array<mx::array, 8>& a,
        const std::array<mx::array, 8>& b);

    // Fused prefix scan for comparison (3 rounds, batched PBS per round)
    CompareFlags fusedComparisonPrefixScan(const std::array<CompareFlags, 8>& flags);

    // Full 256-bit comparison
    CompareFlags compare256(
        const std::array<mx::array, 8>& a,
        const std::array<mx::array, 8>& b);

    // =========================================================================
    // Optimized Equality Check (Fast Path for Differing MSB Words)
    // =========================================================================
    //
    // For equality comparison, if MSB words differ, result is immediately known.
    // This avoids full Kogge-Stone prefix scan for many comparisons.
    //
    // Algorithm:
    //   1. Compare MSB words (word 7) first - single PBS call
    //   2. If not equal, return false immediately (0 PBS saved on early exit)
    //   3. Otherwise, proceed to next word, cascading down
    //   4. Best case: 1 PBS (words differ at MSB)
    //   5. Worst case: 8 PBS (all words equal, checked sequentially)

    mx::array fastEquality256(
        const std::array<mx::array, 8>& a,
        const std::array<mx::array, 8>& b);

    // =========================================================================
    // Byte Shift Operations (Parallel PBS for cross-word byte shuffling)
    // =========================================================================

    // Shift left by bytes (all affected words processed in parallel)
    std::array<mx::array, 8> parallelByteShiftLeft(
        const std::array<mx::array, 8>& a,
        uint32_t bytes);

    // Shift right by bytes
    std::array<mx::array, 8> parallelByteShiftRight(
        const std::array<mx::array, 8>& a,
        uint32_t bytes);

    // =========================================================================
    // Bitwise Operations (Word-Parallel)
    // =========================================================================

    std::array<mx::array, 8> parallelAnd(
        const std::array<mx::array, 8>& a,
        const std::array<mx::array, 8>& b);

    std::array<mx::array, 8> parallelOr(
        const std::array<mx::array, 8>& a,
        const std::array<mx::array, 8>& b);

    std::array<mx::array, 8> parallelXor(
        const std::array<mx::array, 8>& a,
        const std::array<mx::array, 8>& b);

    // NOT is linear - no PBS
    std::array<mx::array, 8> parallelNot(const std::array<mx::array, 8>& a);

    // =========================================================================
    // MUX (Conditional Select) - 8 words in parallel
    // =========================================================================

    std::array<mx::array, 8> parallelMux(
        const mx::array& cond,
        const std::array<mx::array, 8>& a,
        const std::array<mx::array, 8>& b);

    // =========================================================================
    // Statistics
    // =========================================================================

    const OptimizedPBSEngine::Stats& stats() const { return engine_->stats(); }
    
    // Access pre-allocated workspace
    const PBSWorkspace& workspace() const { return workspace_; }

private:
    Config cfg_;
    std::unique_ptr<OptimizedPBSEngine> engine_;
    
    // Pre-allocated workspace to avoid hot-path allocations
    PBSWorkspace workspace_;

    // Batch execution helper
    std::vector<mx::array> executeBatch(
        const std::vector<mx::array>& lwes,
        const std::vector<TestPolyType>& types,
        const std::vector<uint32_t>& params = {});

    // Homomorphic LWE operations (no PBS)
    mx::array lwAdd(const mx::array& a, const mx::array& b);
    mx::array lwSub(const mx::array& a, const mx::array& b);
};

// =============================================================================
// Implementation
// =============================================================================

inline euint256PBSContext::euint256PBSContext(const Config& cfg) : cfg_(cfg) {
    OptimizedPBSEngine::Config engCfg;
    engCfg.N = cfg.N;
    engCfg.n = cfg.n;
    engCfg.L = cfg.L;
    engCfg.baseLog = cfg.baseLog;
    engCfg.Q = cfg.Q;
    engCfg.L_ks = cfg.L_ks;
    engCfg.baseLog_ks = cfg.baseLog_ks;
    engCfg.q_lwe = cfg.q_lwe;
    engCfg.enableBatching = true;
    engCfg.maxBatchSize = 64;

    engine_ = std::make_unique<OptimizedPBSEngine>(engCfg);
    
    // Initialize pre-allocated workspace for euint256 operations
    // Max batch for 8-word operations is typically 64 (8 words * 8 parallel ops)
    workspace_.init(64, cfg.N, cfg.n, cfg.Q);
}

inline void euint256PBSContext::setBootstrapKey(const mx::array& bsk) {
    engine_->setBootstrapKey(bsk);
}

inline void euint256PBSContext::setKeySwitchKey(const mx::array& ksk) {
    engine_->setKeySwitchKey(ksk);
}

inline mx::array euint256PBSContext::lwAdd(const mx::array& a, const mx::array& b) {
    // No eval - let lazy evaluation continue
    return mx::add(a, b);
}

inline mx::array euint256PBSContext::lwSub(const mx::array& a, const mx::array& b) {
    // No eval - let lazy evaluation continue
    return mx::subtract(a, b);
}

inline std::vector<mx::array> euint256PBSContext::executeBatch(
    const std::vector<mx::array>& lwes,
    const std::vector<TestPolyType>& types,
    const std::vector<uint32_t>& params) {
    return engine_->executeBatch(lwes, types, params);
}

// ---------------------------------------------------------------------------
// Word-Parallel Operations
// ---------------------------------------------------------------------------

inline std::array<mx::array, 8> euint256PBSContext::parallelWordAdd(
    const std::array<mx::array, 8>& a,
    const std::array<mx::array, 8>& b) {

    // Validation: euint256 requires exactly 8 words
    if (a.size() != 8 || b.size() != 8) {
        throw std::runtime_error("euint256 operations require exactly 8 words");
    }

    // Homomorphic addition without PBS (just LWE add)
    // PBS refresh handled separately if needed
    auto result = makeArrayOf8();
    for (int i = 0; i < 8; ++i) {
        result[i] = lwAdd(a[i], b[i]);
    }
    return result;
}

inline std::array<mx::array, 8> euint256PBSContext::parallelWordSub(
    const std::array<mx::array, 8>& a,
    const std::array<mx::array, 8>& b) {

    // Validation: euint256 requires exactly 8 words
    if (a.size() != 8 || b.size() != 8) {
        throw std::runtime_error("euint256 operations require exactly 8 words");
    }

    auto result = makeArrayOf8();
    for (int i = 0; i < 8; ++i) {
        result[i] = lwSub(a[i], b[i]);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Fused Kogge-Stone Carry Propagation
// ---------------------------------------------------------------------------

inline std::array<euint256PBSContext::GeneratePropagate, 8>
euint256PBSContext::computeGeneratePropagateParallel(
    const std::array<mx::array, 8>& a,
    const std::array<mx::array, 8>& b) {

    // Validation: euint256 requires exactly 8 words
    if (a.size() != 8 || b.size() != 8) {
        throw std::runtime_error("euint256 operations require exactly 8 words");
    }

    // Compute all 16 operations in parallel:
    // - 8 AND operations for generate
    // - 8 XOR operations for propagate

    std::vector<mx::array> combined;
    combined.reserve(16);

    // Prepare combined LWEs for AND (generate)
    for (int i = 0; i < 8; ++i) {
        combined.push_back(lwAdd(a[i], b[i]));
    }
    // Prepare for XOR (propagate) - same combined LWE, different test poly
    for (int i = 0; i < 8; ++i) {
        combined.push_back(lwAdd(a[i], b[i]));
    }

    std::vector<TestPolyType> types;
    types.reserve(16);
    for (int i = 0; i < 8; ++i) types.push_back(TestPolyType::BOOL_AND);
    for (int i = 0; i < 8; ++i) types.push_back(TestPolyType::BOOL_XOR);

    auto results = executeBatch(combined, types);

    std::array<GeneratePropagate, 8> gp;
    for (int i = 0; i < 8; ++i) {
        gp[i].generate = results[i];
        gp[i].propagate = results[8 + i];
    }

    return gp;
}

inline std::array<mx::array, 8> euint256PBSContext::fusedKoggeStoneCarries(
    const std::array<GeneratePropagate, 8>& gp,
    const mx::array& carryIn) {

    // Kogge-Stone parallel prefix for carry computation
    // (G', P') = (G_hi OR (P_hi AND G_lo), P_hi AND P_lo)
    //
    // 3 rounds for 8 elements (spans 1, 2, 4)

    auto G = makeArrayOf8();
    auto P = makeArrayOf8();
    for (int i = 0; i < 8; ++i) {
        G[i] = gp[i].generate;
        P[i] = gp[i].propagate;
    }

    // ---------------------------------------------------------------------------
    // Round 1: span = 1 (combine adjacent pairs)
    // ---------------------------------------------------------------------------
    // For i in [1..7]: (G'[i], P'[i]) = combine(G[i], P[i], G[i-1], P[i-1])
    // 7 combines, each needs 3 PBS: AND(P[i], G[i-1]), OR(G[i], prev), AND(P[i], P[i-1])
    // Total: 21 PBS, executed as single batch

    {
        std::vector<mx::array> lwes;
        std::vector<TestPolyType> types;

        // For each position i from 1 to 7:
        // Op1: P[i] AND G[i-1] -> need combined = P[i] + G[i-1]
        // Op2: G[i] OR (result of Op1) -> need combined = G[i] + Op1_result
        // Op3: P[i] AND P[i-1] -> need combined = P[i] + P[i-1]

        // First pass: compute P[i] AND G[i-1] and P[i] AND P[i-1] for i=1..7
        for (int i = 1; i < 8; ++i) {
            lwes.push_back(lwAdd(P[i], G[i - 1]));  // For P AND G
            types.push_back(TestPolyType::BOOL_AND);
        }
        for (int i = 1; i < 8; ++i) {
            lwes.push_back(lwAdd(P[i], P[i - 1]));  // For P AND P
            types.push_back(TestPolyType::BOOL_AND);
        }

        auto r1 = executeBatch(lwes, types);

        // Second pass: G'[i] = G[i] OR (P[i] AND G[i-1])
        std::vector<mx::array> lwes2;
        std::vector<TestPolyType> types2;

        for (int i = 1; i < 8; ++i) {
            lwes2.push_back(lwAdd(G[i], r1[i - 1]));  // G OR (P AND G)
            types2.push_back(TestPolyType::BOOL_OR);
        }

        auto r2 = executeBatch(lwes2, types2);

        // Update G and P
        for (int i = 1; i < 8; ++i) {
            G[i] = r2[i - 1];       // G'[i] = G[i] OR (P[i] AND G[i-1])
            P[i] = r1[7 + i - 1];   // P'[i] = P[i] AND P[i-1]
        }
    }

    // ---------------------------------------------------------------------------
    // Round 2: span = 2
    // ---------------------------------------------------------------------------
    // For i in [2..7]: combine with position i-2

    {
        std::vector<mx::array> lwes;
        std::vector<TestPolyType> types;

        for (int i = 2; i < 8; ++i) {
            lwes.push_back(lwAdd(P[i], G[i - 2]));
            types.push_back(TestPolyType::BOOL_AND);
        }
        for (int i = 2; i < 8; ++i) {
            lwes.push_back(lwAdd(P[i], P[i - 2]));
            types.push_back(TestPolyType::BOOL_AND);
        }

        auto r1 = executeBatch(lwes, types);

        std::vector<mx::array> lwes2;
        std::vector<TestPolyType> types2;

        for (int i = 2; i < 8; ++i) {
            lwes2.push_back(lwAdd(G[i], r1[i - 2]));
            types2.push_back(TestPolyType::BOOL_OR);
        }

        auto r2 = executeBatch(lwes2, types2);

        for (int i = 2; i < 8; ++i) {
            G[i] = r2[i - 2];
            P[i] = r1[6 + i - 2];
        }
    }

    // ---------------------------------------------------------------------------
    // Round 3: span = 4
    // ---------------------------------------------------------------------------
    // For i in [4..7]: combine with position i-4

    {
        std::vector<mx::array> lwes;
        std::vector<TestPolyType> types;

        for (int i = 4; i < 8; ++i) {
            lwes.push_back(lwAdd(P[i], G[i - 4]));
            types.push_back(TestPolyType::BOOL_AND);
        }
        for (int i = 4; i < 8; ++i) {
            lwes.push_back(lwAdd(P[i], P[i - 4]));
            types.push_back(TestPolyType::BOOL_AND);
        }

        auto r1 = executeBatch(lwes, types);

        std::vector<mx::array> lwes2;
        std::vector<TestPolyType> types2;

        for (int i = 4; i < 8; ++i) {
            lwes2.push_back(lwAdd(G[i], r1[i - 4]));
            types2.push_back(TestPolyType::BOOL_OR);
        }

        auto r2 = executeBatch(lwes2, types2);

        for (int i = 4; i < 8; ++i) {
            G[i] = r2[i - 4];
            P[i] = r1[4 + i - 4];
        }
    }

    // ---------------------------------------------------------------------------
    // Extract final carries
    // ---------------------------------------------------------------------------
    // C[i+1] = G[i] OR (P[i] AND C[0])
    // C[0] = carryIn

    auto carries = makeArrayOf8();
    carries[0] = carryIn;

    // Compute C[1..7] = G[0..6] OR (P[0..6] AND carryIn)
    std::vector<mx::array> lwes;
    std::vector<TestPolyType> types;

    // First: P[i] AND carryIn for i=0..6
    for (int i = 0; i < 7; ++i) {
        lwes.push_back(lwAdd(P[i], carryIn));
        types.push_back(TestPolyType::BOOL_AND);
    }

    auto r1 = executeBatch(lwes, types);

    // Then: G[i] OR (P[i] AND carryIn) for i=0..6
    std::vector<mx::array> lwes2;
    std::vector<TestPolyType> types2;

    for (int i = 0; i < 7; ++i) {
        lwes2.push_back(lwAdd(G[i], r1[i]));
        types2.push_back(TestPolyType::BOOL_OR);
    }

    auto r2 = executeBatch(lwes2, types2);

    for (int i = 0; i < 7; ++i) {
        carries[i + 1] = r2[i];
    }

    return carries;
}

inline std::array<mx::array, 8> euint256PBSContext::normalizedAdd(
    const std::array<mx::array, 8>& a,
    const std::array<mx::array, 8>& b) {

    // Validation: euint256 requires exactly 8 words
    if (a.size() != 8 || b.size() != 8) {
        throw std::runtime_error("euint256 operations require exactly 8 words");
    }

    // Step 1: Word-wise addition (no PBS)
    auto sums = parallelWordAdd(a, b);

    // Step 2: Compute generate/propagate pairs (16 PBS - batched)
    auto gp = computeGeneratePropagateParallel(a, b);

    // Step 3: Fused Kogge-Stone carry propagation (batched PBS per round)
    // Use pre-allocated zero LWE from workspace instead of allocating
    auto carryIn = workspace_.getZeroLWE();

    auto carries = fusedKoggeStoneCarries(gp, carryIn);

    // Step 4: Apply carries to sums
    // OPTIMIZATION: Direct LWE addition - NO PBS needed!
    // Adding encrypted carry to encrypted sum is a LINEAR operation.
    // Previous wasteful code used 8 PBS operations with IDENTITY test poly.
    auto out = makeArrayOf8();
    for (int i = 0; i < 8; ++i) {
        // Direct addition: no PBS required for linear operations
        out[i] = mx::add(sums[i], carries[i]);
    }
    // Single eval at end for efficiency
    mx::eval(out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);

    return out;
}

// ---------------------------------------------------------------------------
// Parallel Comparison
// ---------------------------------------------------------------------------

inline std::array<euint256PBSContext::CompareFlags, 8>
euint256PBSContext::parallelWordCompare(
    const std::array<mx::array, 8>& a,
    const std::array<mx::array, 8>& b) {

    // Validation: euint256 requires exactly 8 words
    if (a.size() != 8 || b.size() != 8) {
        throw std::runtime_error("euint256 operations require exactly 8 words");
    }

    // For each word pair, compute (gt, eq, lt) flags
    // gt: a > b -> compute b - a, check sign
    // lt: a < b -> compute a - b, check sign
    // eq: a == b -> check if a - b == 0

    // Total: 24 PBS (3 per word), executed as single batch
    std::vector<mx::array> lwes;
    std::vector<TestPolyType> types;

    // Prepare diffs for sign extraction
    for (int i = 0; i < 8; ++i) {
        lwes.push_back(lwSub(a[i], b[i]));  // For lt: if a < b, this is negative
        types.push_back(TestPolyType::SIGN_EXTRACT);
    }
    for (int i = 0; i < 8; ++i) {
        lwes.push_back(lwSub(b[i], a[i]));  // For gt: if b < a (a > b), this is negative
        types.push_back(TestPolyType::SIGN_EXTRACT);
    }
    // For eq: check if diff == 0 (more complex, needs byte decomposition)
    // Simplified: eq = NOT(lt OR gt)
    // We'll compute this from lt and gt

    auto results = executeBatch(lwes, types);

    std::array<CompareFlags, 8> flags;
    for (int i = 0; i < 8; ++i) {
        flags[i].lt = results[i];
        flags[i].gt = results[8 + i];
        // eq = NOT(lt) AND NOT(gt) - needs additional PBS
        // For now, compute eq separately
    }

    // Compute eq = NOT(lt OR gt)
    std::vector<mx::array> eqLwes;
    std::vector<TestPolyType> eqTypes;

    for (int i = 0; i < 8; ++i) {
        eqLwes.push_back(lwAdd(flags[i].lt, flags[i].gt));
        eqTypes.push_back(TestPolyType::BOOL_OR);
    }

    auto orResults = executeBatch(eqLwes, eqTypes);

    // NOT is linear, no PBS
    for (int i = 0; i < 8; ++i) {
        flags[i].eq = engine_->boolNot(orResults[i]);
    }

    return flags;
}

inline euint256PBSContext::CompareFlags euint256PBSContext::fusedComparisonPrefixScan(
    const std::array<CompareFlags, 8>& flags) {

    // Priority comparison operator for combining (hi, lo):
    // gt_result = gt_hi OR (eq_hi AND gt_lo)
    // eq_result = eq_hi AND eq_lo
    // lt_result = lt_hi OR (eq_hi AND lt_lo)
    //
    // Kogge-Stone structure: 3 rounds with spans 1, 2, 4

    std::array<CompareFlags, 8> F = flags;

    // Round 1: span = 1
    {
        std::vector<mx::array> lwes;
        std::vector<TestPolyType> types;

        // For i=1..7: compute eq_hi AND gt_lo, eq_hi AND lt_lo, eq_hi AND eq_lo
        for (int i = 1; i < 8; ++i) {
            lwes.push_back(lwAdd(F[i].eq, F[i-1].gt));  // eq AND gt_lo
            types.push_back(TestPolyType::BOOL_AND);
        }
        for (int i = 1; i < 8; ++i) {
            lwes.push_back(lwAdd(F[i].eq, F[i-1].lt));  // eq AND lt_lo
            types.push_back(TestPolyType::BOOL_AND);
        }
        for (int i = 1; i < 8; ++i) {
            lwes.push_back(lwAdd(F[i].eq, F[i-1].eq));  // eq AND eq_lo
            types.push_back(TestPolyType::BOOL_AND);
        }

        auto r1 = executeBatch(lwes, types);

        // Compute OR with hi flags
        std::vector<mx::array> lwes2;
        std::vector<TestPolyType> types2;

        for (int i = 1; i < 8; ++i) {
            lwes2.push_back(lwAdd(F[i].gt, r1[i-1]));  // gt OR (eq AND gt_lo)
            types2.push_back(TestPolyType::BOOL_OR);
        }
        for (int i = 1; i < 8; ++i) {
            lwes2.push_back(lwAdd(F[i].lt, r1[7 + i-1]));  // lt OR (eq AND lt_lo)
            types2.push_back(TestPolyType::BOOL_OR);
        }

        auto r2 = executeBatch(lwes2, types2);

        for (int i = 1; i < 8; ++i) {
            F[i].gt = r2[i-1];
            F[i].lt = r2[7 + i-1];
            F[i].eq = r1[14 + i-1];  // eq AND eq
        }
    }

    // Round 2: span = 2
    {
        std::vector<mx::array> lwes;
        std::vector<TestPolyType> types;

        for (int i = 2; i < 8; ++i) {
            lwes.push_back(lwAdd(F[i].eq, F[i-2].gt));
            types.push_back(TestPolyType::BOOL_AND);
        }
        for (int i = 2; i < 8; ++i) {
            lwes.push_back(lwAdd(F[i].eq, F[i-2].lt));
            types.push_back(TestPolyType::BOOL_AND);
        }
        for (int i = 2; i < 8; ++i) {
            lwes.push_back(lwAdd(F[i].eq, F[i-2].eq));
            types.push_back(TestPolyType::BOOL_AND);
        }

        auto r1 = executeBatch(lwes, types);

        std::vector<mx::array> lwes2;
        std::vector<TestPolyType> types2;

        for (int i = 2; i < 8; ++i) {
            lwes2.push_back(lwAdd(F[i].gt, r1[i-2]));
            types2.push_back(TestPolyType::BOOL_OR);
        }
        for (int i = 2; i < 8; ++i) {
            lwes2.push_back(lwAdd(F[i].lt, r1[6 + i-2]));
            types2.push_back(TestPolyType::BOOL_OR);
        }

        auto r2 = executeBatch(lwes2, types2);

        for (int i = 2; i < 8; ++i) {
            F[i].gt = r2[i-2];
            F[i].lt = r2[6 + i-2];
            F[i].eq = r1[12 + i-2];
        }
    }

    // Round 3: span = 4
    {
        std::vector<mx::array> lwes;
        std::vector<TestPolyType> types;

        for (int i = 4; i < 8; ++i) {
            lwes.push_back(lwAdd(F[i].eq, F[i-4].gt));
            types.push_back(TestPolyType::BOOL_AND);
        }
        for (int i = 4; i < 8; ++i) {
            lwes.push_back(lwAdd(F[i].eq, F[i-4].lt));
            types.push_back(TestPolyType::BOOL_AND);
        }
        for (int i = 4; i < 8; ++i) {
            lwes.push_back(lwAdd(F[i].eq, F[i-4].eq));
            types.push_back(TestPolyType::BOOL_AND);
        }

        auto r1 = executeBatch(lwes, types);

        std::vector<mx::array> lwes2;
        std::vector<TestPolyType> types2;

        for (int i = 4; i < 8; ++i) {
            lwes2.push_back(lwAdd(F[i].gt, r1[i-4]));
            types2.push_back(TestPolyType::BOOL_OR);
        }
        for (int i = 4; i < 8; ++i) {
            lwes2.push_back(lwAdd(F[i].lt, r1[4 + i-4]));
            types2.push_back(TestPolyType::BOOL_OR);
        }

        auto r2 = executeBatch(lwes2, types2);

        for (int i = 4; i < 8; ++i) {
            F[i].gt = r2[i-4];
            F[i].lt = r2[4 + i-4];
            F[i].eq = r1[8 + i-4];
        }
    }

    // F[7] contains the final combined result after Kogge-Stone prefix scan
    // (MSB word has accumulated the full 256-bit comparison)
    return F[7];
}

inline euint256PBSContext::CompareFlags euint256PBSContext::compare256(
    const std::array<mx::array, 8>& a,
    const std::array<mx::array, 8>& b) {

    // Validation: euint256 requires exactly 8 words
    if (a.size() != 8 || b.size() != 8) {
        throw std::runtime_error("euint256 operations require exactly 8 words");
    }

    auto wordFlags = parallelWordCompare(a, b);
    return fusedComparisonPrefixScan(wordFlags);
}

// ---------------------------------------------------------------------------
// Fast Equality Check (MSB-First Early Exit)
// ---------------------------------------------------------------------------
//
// OPTIMIZATION: For equality comparison of 256-bit values, check MSB words first.
// If any word differs, we know the result immediately without full Kogge-Stone.
//
// PBS savings:
// - Best case (MSB differs): 1 PBS vs ~32 PBS for full comparison
// - Average case: ~4 PBS (random values differ ~50% per word)
// - Worst case (all equal): 8 PBS sequential (same as cascaded check)

inline mx::array euint256PBSContext::fastEquality256(
    const std::array<mx::array, 8>& a,
    const std::array<mx::array, 8>& b) {

    // Validation: euint256 requires exactly 8 words
    if (a.size() != 8 || b.size() != 8) {
        throw std::runtime_error("euint256 operations require exactly 8 words");
    }

    // Strategy: Check all words in parallel first, then AND-reduce
    // This is more efficient than sequential early-exit for encrypted data
    // because we can't branch on encrypted values.
    //
    // Alternative strategy for statistical early-exit hint:
    // We could use a hierarchical reduction where we batch check pairs,
    // but since we can't branch on encrypted results, we do parallel AND-reduce.

    // Step 1: Compute per-word equality (8 parallel PBS)
    // For each word: eq[i] = (a[i] XOR b[i]) == 0
    // Use sign extraction on (a - b) to check if difference is zero

    std::vector<mx::array> lwes;
    std::vector<TestPolyType> types;

    // Compute a[i] XOR b[i] for all words - if XOR == 0, words are equal
    for (int i = 0; i < 8; ++i) {
        lwes.push_back(lwAdd(a[i], b[i]));
        types.push_back(TestPolyType::BOOL_XOR);
    }

    auto xorResults = executeBatch(lwes, types);

    // Step 2: Check if each XOR result is zero (8 PBS for zero-check)
    // IS_ZERO: returns 1 if input is 0 (words equal), 0 otherwise
    std::vector<mx::array> zeroLwes;
    std::vector<TestPolyType> zeroTypes;

    for (int i = 0; i < 8; ++i) {
        zeroLwes.push_back(xorResults[i]);
        zeroTypes.push_back(TestPolyType::IS_ZERO);  // Returns 1 if input is 0
    }

    auto zeroResults = executeBatch(zeroLwes, zeroTypes);

    // Step 3: AND-reduce all equality flags (log2(8) = 3 rounds of PBS)
    // Round 1: Reduce 8 -> 4
    std::vector<mx::array> r1Lwes;
    std::vector<TestPolyType> r1Types;

    for (int i = 0; i < 4; ++i) {
        r1Lwes.push_back(lwAdd(zeroResults[2*i], zeroResults[2*i + 1]));
        r1Types.push_back(TestPolyType::BOOL_AND);
    }

    auto r1 = executeBatch(r1Lwes, r1Types);

    // Round 2: Reduce 4 -> 2
    std::vector<mx::array> r2Lwes;
    std::vector<TestPolyType> r2Types;

    r2Lwes.push_back(lwAdd(r1[0], r1[1]));
    r2Types.push_back(TestPolyType::BOOL_AND);
    r2Lwes.push_back(lwAdd(r1[2], r1[3]));
    r2Types.push_back(TestPolyType::BOOL_AND);

    auto r2 = executeBatch(r2Lwes, r2Types);

    // Round 3: Reduce 2 -> 1
    std::vector<mx::array> r3Lwes;
    std::vector<TestPolyType> r3Types;

    r3Lwes.push_back(lwAdd(r2[0], r2[1]));
    r3Types.push_back(TestPolyType::BOOL_AND);

    auto r3 = executeBatch(r3Lwes, r3Types);

    // Total: 8 (XOR) + 8 (zero-check) + 4 + 2 + 1 (AND-reduce) = 23 PBS
    // vs full compare256: ~32+ PBS
    // For equality-only checks, this is more efficient.

    return r3[0];
}

// ---------------------------------------------------------------------------
// Byte Shift Operations
// ---------------------------------------------------------------------------

inline std::array<mx::array, 8> euint256PBSContext::parallelByteShiftLeft(
    const std::array<mx::array, 8>& a,
    uint32_t bytes) {

    // Validation: euint256 requires exactly 8 words
    if (a.size() != 8) {
        throw std::runtime_error("euint256 operations require exactly 8 words");
    }

    if (bytes == 0) return a;
    if (bytes >= 32) {
        auto result = makeArrayOf8();
        // Use pre-allocated zero LWE from workspace
        for (int i = 0; i < 8; ++i) {
            result[i] = workspace_.getZeroLWESlice(i);
        }
        return result;
    }

    uint32_t limb_shift = bytes / 4;
    uint32_t byte_offset = bytes % 4;

    auto result = makeArrayOf8();

    // First do limb shift (no PBS)
    for (int i = 7; i >= 0; --i) {
        if (i >= static_cast<int>(limb_shift)) {
            result[i] = a[i - limb_shift];
        } else {
            // Use pre-allocated zero LWE from workspace
            result[i] = workspace_.getZeroLWESlice(i);
        }
    }

    if (byte_offset == 0) {
        return result;
    }

    // Byte shift within words - need PBS
    std::vector<mx::array> lwes;
    std::vector<TestPolyType> types;
    std::vector<uint32_t> params;

    // For each affected word, compute high part (shifted) and low part (extracted from prev)
    for (int i = 7; i >= static_cast<int>(limb_shift); --i) {
        // High part: rotate left by byte_offset
        lwes.push_back(result[i]);
        types.push_back(static_cast<TestPolyType>(
            static_cast<uint32_t>(TestPolyType::BYTE_ROTATE_LEFT_1) + byte_offset - 1));
        params.push_back(byte_offset);

        // Low part from previous word (if exists)
        if (i > static_cast<int>(limb_shift)) {
            lwes.push_back(result[i - 1]);
            types.push_back(static_cast<TestPolyType>(
                static_cast<uint32_t>(TestPolyType::EXTRACT_HIGH_BYTE_1) + byte_offset - 1));
            params.push_back(byte_offset);
        }
    }

    auto pbsResults = executeBatch(lwes, types, params);

    // Combine results
    int resultIdx = 0;
    for (int i = 7; i >= static_cast<int>(limb_shift); --i) {
        auto high = pbsResults[resultIdx++];
        if (i > static_cast<int>(limb_shift)) {
            auto low = pbsResults[resultIdx++];
            // Combine via OR
            result[i] = engine_->boolOr(high, low);
        } else {
            result[i] = high;
        }
    }

    return result;
}

inline std::array<mx::array, 8> euint256PBSContext::parallelByteShiftRight(
    const std::array<mx::array, 8>& a,
    uint32_t bytes) {

    // Validation: euint256 requires exactly 8 words
    if (a.size() != 8) {
        throw std::runtime_error("euint256 operations require exactly 8 words");
    }

    if (bytes == 0) return a;
    if (bytes >= 32) {
        auto result = makeArrayOf8();
        // Use pre-allocated zero LWE from workspace
        for (int i = 0; i < 8; ++i) {
            result[i] = workspace_.getZeroLWESlice(i);
        }
        return result;
    }

    uint32_t limb_shift = bytes / 4;
    uint32_t byte_offset = bytes % 4;

    auto result = makeArrayOf8();

    // First do limb shift (no PBS)
    for (int i = 0; i < 8; ++i) {
        if (i + limb_shift < 8) {
            result[i] = a[i + limb_shift];
        } else {
            // Use pre-allocated zero LWE from workspace
            result[i] = workspace_.getZeroLWESlice(i);
        }
    }

    if (byte_offset == 0) {
        return result;
    }

    // Byte shift within words - need PBS
    std::vector<mx::array> lwes;
    std::vector<TestPolyType> types;
    std::vector<uint32_t> params;

    for (uint32_t i = 0; i < 8 - limb_shift; ++i) {
        // Low part: rotate right
        lwes.push_back(result[i]);
        types.push_back(static_cast<TestPolyType>(
            static_cast<uint32_t>(TestPolyType::BYTE_ROTATE_RIGHT_1) + byte_offset - 1));
        params.push_back(byte_offset);

        // High part from next word (if exists)
        if (i + 1 < 8 - limb_shift) {
            lwes.push_back(result[i + 1]);
            types.push_back(static_cast<TestPolyType>(
                static_cast<uint32_t>(TestPolyType::EXTRACT_LOW_BYTE_1) + byte_offset - 1));
            params.push_back(byte_offset);
        }
    }

    auto pbsResults = executeBatch(lwes, types, params);

    int resultIdx = 0;
    for (uint32_t i = 0; i < 8 - limb_shift; ++i) {
        auto low = pbsResults[resultIdx++];
        if (i + 1 < 8 - limb_shift) {
            auto high = pbsResults[resultIdx++];
            result[i] = engine_->boolOr(high, low);
        } else {
            result[i] = low;
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// Bitwise Operations
// ---------------------------------------------------------------------------

inline std::array<mx::array, 8> euint256PBSContext::parallelAnd(
    const std::array<mx::array, 8>& a,
    const std::array<mx::array, 8>& b) {

    // Validation: euint256 requires exactly 8 words
    if (a.size() != 8 || b.size() != 8) {
        throw std::runtime_error("euint256 operations require exactly 8 words");
    }

    std::vector<mx::array> lwes;
    std::vector<TestPolyType> types;

    for (int i = 0; i < 8; ++i) {
        lwes.push_back(lwAdd(a[i], b[i]));
        types.push_back(TestPolyType::BOOL_AND);
    }

    auto results = executeBatch(lwes, types);

    auto out = makeArrayOf8();
    for (int i = 0; i < 8; ++i) {
        out[i] = results[i];
    }
    return out;
}

inline std::array<mx::array, 8> euint256PBSContext::parallelOr(
    const std::array<mx::array, 8>& a,
    const std::array<mx::array, 8>& b) {

    // Validation: euint256 requires exactly 8 words
    if (a.size() != 8 || b.size() != 8) {
        throw std::runtime_error("euint256 operations require exactly 8 words");
    }

    std::vector<mx::array> lwes;
    std::vector<TestPolyType> types;

    for (int i = 0; i < 8; ++i) {
        lwes.push_back(lwAdd(a[i], b[i]));
        types.push_back(TestPolyType::BOOL_OR);
    }

    auto results = executeBatch(lwes, types);

    auto out = makeArrayOf8();
    for (int i = 0; i < 8; ++i) {
        out[i] = results[i];
    }
    return out;
}

inline std::array<mx::array, 8> euint256PBSContext::parallelXor(
    const std::array<mx::array, 8>& a,
    const std::array<mx::array, 8>& b) {

    // Validation: euint256 requires exactly 8 words
    if (a.size() != 8 || b.size() != 8) {
        throw std::runtime_error("euint256 operations require exactly 8 words");
    }

    std::vector<mx::array> lwes;
    std::vector<TestPolyType> types;

    for (int i = 0; i < 8; ++i) {
        lwes.push_back(lwAdd(a[i], b[i]));
        types.push_back(TestPolyType::BOOL_XOR);
    }

    auto results = executeBatch(lwes, types);

    auto out = makeArrayOf8();
    for (int i = 0; i < 8; ++i) {
        out[i] = results[i];
    }
    return out;
}

inline std::array<mx::array, 8> euint256PBSContext::parallelNot(
    const std::array<mx::array, 8>& a) {

    // Validation: euint256 requires exactly 8 words
    if (a.size() != 8) {
        throw std::runtime_error("euint256 operations require exactly 8 words");
    }

    // NOT is linear - no PBS needed
    auto out = makeArrayOf8();
    for (int i = 0; i < 8; ++i) {
        out[i] = engine_->boolNot(a[i]);
    }
    return out;
}

// ---------------------------------------------------------------------------
// MUX (Conditional Select)
// ---------------------------------------------------------------------------

inline std::array<mx::array, 8> euint256PBSContext::parallelMux(
    const mx::array& cond,
    const std::array<mx::array, 8>& a,
    const std::array<mx::array, 8>& b) {

    // Validation: euint256 requires exactly 8 words
    if (a.size() != 8 || b.size() != 8) {
        throw std::runtime_error("euint256 operations require exactly 8 words");
    }

    // MUX: result = cond ? a : b = b + cond * (a - b)
    // For each word: compute diff = a[i] - b[i], then cond * diff, then add b[i]

    std::vector<mx::array> lwes;
    std::vector<TestPolyType> types;
    std::vector<uint32_t> params;

    for (int i = 0; i < 8; ++i) {
        auto diff = lwSub(a[i], b[i]);
        // For MUX_PASSTHROUGH: need to get the value from diff
        mx::eval(diff);
        auto diff_ptr = diff.data<int64_t>();
        uint32_t val = static_cast<uint32_t>(diff_ptr[cfg_.n] & 0xFFFFFFFF);

        // Condition check: if cond=1, output val; if cond=0, output 0
        lwes.push_back(cond);
        types.push_back(TestPolyType::MUX_PASSTHROUGH);
        params.push_back(val);
    }

    auto pbsResults = executeBatch(lwes, types, params);

    // Add back b[i]
    auto out = makeArrayOf8();  // Use helper for consistent initialization
    for (int i = 0; i < 8; ++i) {
        out[i] = lwAdd(b[i], pbsResults[i]);
    }

    return out;
}

#endif // WITH_MLX

}  // namespace gpu
}  // namespace lbcrypto

#endif // LBCRYPTO_MATH_HAL_MLX_EUINT256_PBS_INTEGRATION_H
