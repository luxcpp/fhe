// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// Lazy Carry Propagation for Radix Integers
//
// Separates "VM semantics" from "FHE core" with explicit normalization
// boundaries. EVM wants exact uint256 semantics, but doing PBS after
// every primitive is expensive.
//
// Key insight: Each limb can accumulate up to carry_bits worth of carry
// before overflow. Track "carry depth" globally and defer PBS until required.
//
// Normalization is required at:
// - Opcode boundaries requiring canonical form (comparisons, div/mod, shifts)
// - Before serialization/proof steps
// - When carry budget would overflow

#ifndef RADIX_LAZY_CARRY_H
#define RADIX_LAZY_CARRY_H

#include "radix/shortint.h"
#include "batch/binfhe-batch.h"
#include <cstdint>
#include <vector>
#include <limits>

namespace lbcrypto {
namespace radix {

/**
 * @brief CarryBudget tracks accumulated carries across limbs
 *
 * For parameters with message_bits=m and carry_bits=c:
 * - Each limb holds values in [0, 2^m - 1] when normalized
 * - Carry space allows values up to [0, 2^(m+c) - 1]
 * - After k additions, max limb value is k * (2^m - 1)
 * - Overflow occurs when k * (2^m - 1) >= 2^(m+c)
 *
 * Solving for k: k_max = floor(2^(m+c) / (2^m - 1))
 *
 * For m=2, c=2: k_max = floor(16 / 3) = 5 operations before overflow
 */
struct CarryBudget {
    // Number of operations since last normalization
    uint32_t ops_since_normalize = 0;

    // Maximum operations before overflow (computed from params)
    uint32_t max_ops = 0;

    // Highest limb index that may have accumulated carries
    // Used for partial normalization optimization
    uint32_t dirty_high_limb = 0;

    // Track operation types for more precise budgeting
    // Addition adds at most (2^m - 1) per op
    // Multiplication can add (2^m - 1)^2 per limb-pair
    enum class OpType : uint8_t {
        NONE,
        ADD,        // +1 to max limb value
        SUB,        // -1 to min (handled via wrap)
        MUL,        // +(2^m - 1) to max per limb
        SCALAR_MUL  // +scalar contribution
    };

    OpType last_op = OpType::NONE;

    // Check if another operation of given type can proceed
    bool CanPerform(OpType op) const {
        if (ops_since_normalize >= max_ops) {
            return false;
        }
        // Multiplication consumes more budget
        if (op == OpType::MUL) {
            return ops_since_normalize + 2 <= max_ops;
        }
        return true;
    }

    // Record that an operation was performed
    void RecordOp(OpType op, uint32_t affected_high_limb = 0) {
        if (op == OpType::MUL) {
            ops_since_normalize += 2;  // Mul consumes 2 budget slots
        } else if (op != OpType::NONE) {
            ops_since_normalize += 1;
        }
        last_op = op;
        if (affected_high_limb > dirty_high_limb) {
            dirty_high_limb = affected_high_limb;
        }
    }

    // Reset after normalization
    void Reset() {
        ops_since_normalize = 0;
        dirty_high_limb = 0;
        last_op = OpType::NONE;
    }

    // Compute max_ops from parameters
    static uint32_t ComputeMaxOps(const ShortIntParams& params) {
        uint64_t total_space = 1ULL << params.total_bits();
        uint64_t max_message = (1ULL << params.message_bits) - 1;

        if (max_message == 0) return std::numeric_limits<uint32_t>::max();

        // For addition: max k where k * max_message < total_space
        uint32_t k = static_cast<uint32_t>(total_space / max_message);

        // Safety margin: use k-1 to ensure no overflow
        return (k > 1) ? k - 1 : 1;
    }
};

/**
 * @brief CarryState tracks per-limb carry accumulation
 *
 * More precise than global budget, but higher overhead.
 * Use for operations where limbs have different carry depths.
 */
struct CarryState {
    // Maximum accumulated value in this limb
    // Starts at (2^m - 1), grows with operations
    uint64_t max_accumulated = 0;

    // Minimum accumulated value (for subtraction tracking)
    // Starts at 0, can go negative (wrapped)
    int64_t min_accumulated = 0;

    // Whether this limb definitely needs normalization
    bool needs_normalize = false;
};

/**
 * @brief NormalizationPolicy determines when to normalize
 */
enum class NormalizationPolicy {
    // Normalize as late as possible (maximize batching)
    LAZY,

    // Normalize at fixed intervals (predictable timing)
    PERIODIC,

    // Normalize immediately when threshold reached
    EAGER,

    // Never auto-normalize (caller responsible)
    MANUAL
};

/**
 * @brief LazyCarryManager handles deferred carry propagation
 *
 * Core class for lazy carry semantics. Wraps a vector of limbs
 * and tracks their carry state, deferring PBS until necessary.
 */
class LazyCarryManager {
public:
    LazyCarryManager() = default;
    explicit LazyCarryManager(const ShortIntParams& params,
                              NormalizationPolicy policy = NormalizationPolicy::LAZY);
    ~LazyCarryManager();

    // Non-copyable but movable
    LazyCarryManager(const LazyCarryManager&) = delete;
    LazyCarryManager& operator=(const LazyCarryManager&) = delete;
    LazyCarryManager(LazyCarryManager&&) noexcept;
    LazyCarryManager& operator=(LazyCarryManager&&) noexcept;

    // ========================================================================
    // Query State
    // ========================================================================

    /**
     * @brief Check if normalization is required
     */
    bool NeedsNormalization() const { return budget_.ops_since_normalize >= budget_.max_ops; }

    /**
     * @brief Check if operation can proceed without normalize
     */
    bool CanPerformOp(CarryBudget::OpType op) const { return budget_.CanPerform(op); }

    /**
     * @brief Get remaining operations before normalize required
     */
    uint32_t RemainingOps() const {
        return (budget_.max_ops > budget_.ops_since_normalize)
               ? budget_.max_ops - budget_.ops_since_normalize
               : 0;
    }

    /**
     * @brief Get maximum operations between normalizations
     */
    uint32_t MaxOpsBeforeNormalize() const { return budget_.max_ops; }

    /**
     * @brief Get current carry budget state
     */
    const CarryBudget& GetBudget() const { return budget_; }

    /**
     * @brief Get parameters
     */
    const ShortIntParams& GetParams() const { return params_; }

    // ========================================================================
    // Record Operations (call after performing lazy ops)
    // ========================================================================

    void RecordAdd(uint32_t high_limb = 0) {
        budget_.RecordOp(CarryBudget::OpType::ADD, high_limb);
        MaybeAutoNormalize();
    }

    void RecordSub(uint32_t high_limb = 0) {
        budget_.RecordOp(CarryBudget::OpType::SUB, high_limb);
        MaybeAutoNormalize();
    }

    void RecordMul(uint32_t high_limb = 0) {
        budget_.RecordOp(CarryBudget::OpType::MUL, high_limb);
        MaybeAutoNormalize();
    }

    void RecordScalarMul(uint32_t high_limb = 0) {
        budget_.RecordOp(CarryBudget::OpType::SCALAR_MUL, high_limb);
        MaybeAutoNormalize();
    }

    // ========================================================================
    // Normalization
    // ========================================================================

    /**
     * @brief Normalize all limbs via batched PBS
     *
     * This is the expensive operation. Batches all dirty limbs into a
     * single PBS dispatch for GPU efficiency.
     *
     * @param cc BinFHE context
     * @param limbs Vector of limbs to normalize (modified in place)
     * @param luts ShortInt LUTs for carry extraction
     */
    void Normalize(BinFHEContext& cc,
                   std::vector<ShortInt>& limbs,
                   const ShortIntLUTs& luts);

    /**
     * @brief Partial normalization up to given limb index
     *
     * Useful when only low limbs are dirty.
     */
    void NormalizeUpTo(BinFHEContext& cc,
                       std::vector<ShortInt>& limbs,
                       uint32_t max_limb,
                       const ShortIntLUTs& luts);

    /**
     * @brief Normalize if required, no-op otherwise
     */
    void NormalizeIfNeeded(BinFHEContext& cc,
                           std::vector<ShortInt>& limbs,
                           const ShortIntLUTs& luts);

    /**
     * @brief Reset carry state (call after external normalization)
     */
    void Reset() { budget_.Reset(); }

    /**
     * @brief Set normalization policy
     */
    void SetPolicy(NormalizationPolicy policy) { policy_ = policy; }

private:
    ShortIntParams params_;
    CarryBudget budget_;
    NormalizationPolicy policy_ = NormalizationPolicy::LAZY;

    // Callback for auto-normalization (set externally)
    std::function<void()> auto_normalize_callback_;

    void MaybeAutoNormalize() {
        if (policy_ == NormalizationPolicy::EAGER && NeedsNormalization()) {
            if (auto_normalize_callback_) {
                auto_normalize_callback_();
            }
        }
    }
};

// ============================================================================
// Lazy Arithmetic Operations
// ============================================================================
// These operations defer carry propagation. Caller must track budget
// and call Normalize() when required.

/**
 * @brief Lazy addition without carry propagation
 *
 * Simply adds limb values. Result may exceed message space but
 * must stay within total_bits space.
 *
 * @param a First operand limbs
 * @param b Second operand limbs
 * @param result Output limbs (sum without carry propagation)
 * @param cc BinFHE context
 */
void LazyAdd(const std::vector<ShortInt>& a,
             const std::vector<ShortInt>& b,
             std::vector<ShortInt>& result,
             BinFHEContext& cc);

/**
 * @brief Lazy subtraction without borrow propagation
 *
 * Computes a - b. Result may wrap around within total_bits space.
 */
void LazySub(const std::vector<ShortInt>& a,
             const std::vector<ShortInt>& b,
             std::vector<ShortInt>& result,
             BinFHEContext& cc);

/**
 * @brief Lazy scalar addition
 *
 * Adds plaintext scalar to encrypted value.
 */
void LazyAddScalar(const std::vector<ShortInt>& a,
                   uint64_t scalar,
                   std::vector<ShortInt>& result,
                   const ShortIntParams& params,
                   BinFHEContext& cc);

/**
 * @brief Lazy scalar multiplication
 *
 * Multiplies by plaintext scalar. May consume extra carry budget.
 */
void LazyMulScalar(const std::vector<ShortInt>& a,
                   uint64_t scalar,
                   std::vector<ShortInt>& result,
                   const ShortIntParams& params,
                   BinFHEContext& cc);

// ============================================================================
// Batched Normalization
// ============================================================================

/**
 * @brief Normalize multiple RadixInts in a single batch
 *
 * For GPU efficiency, collects all dirty limbs across multiple
 * RadixInts and dispatches a single batched PBS.
 *
 * @param cc BinFHE context
 * @param limb_vectors Vector of limb vectors to normalize
 * @param managers Corresponding carry managers
 * @param luts ShortInt LUTs
 */
void BatchNormalize(BinFHEContext& cc,
                    std::vector<std::vector<ShortInt>*>& limb_vectors,
                    std::vector<LazyCarryManager*>& managers,
                    const ShortIntLUTs& luts);

/**
 * @brief Build carry extraction and propagation LUTs
 *
 * Creates specialized LUTs for normalization:
 * - extract_msg: f(x) = x mod 2^m
 * - extract_carry: f(x) = x >> m
 */
struct NormalizationLUTs {
    std::vector<NativeInteger> extract_message;  // x mod 2^m
    std::vector<NativeInteger> extract_carry;    // x >> m
    std::vector<NativeInteger> add_with_carry;   // For carry chain

    explicit NormalizationLUTs(const ShortIntParams& params);
};

} // namespace radix
} // namespace lbcrypto

#endif // RADIX_LAZY_CARRY_H
