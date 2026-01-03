// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// Batch APIs for OpenFHE BinFHE - Required for GPU throughput
// These APIs allow processing multiple ciphertexts in a single call,
// enabling GPU saturation for fhEVM workloads.

#ifndef BINFHE_BATCH_H
#define BINFHE_BATCH_H

#include "binfhecontext.h"
#include <vector>
#include <cstdint>
#include <functional>

namespace lux::fhe {

// Batch processing flags
enum BatchFlags : uint32_t {
    BATCH_DEFAULT = 0,
    BATCH_ASYNC = 1 << 0,           // Return immediately, result available later
    BATCH_INPLACE = 1 << 1,         // Modify input ciphertexts in place
    BATCH_NO_BOOTSTRAP = 1 << 2,    // Skip bootstrapping (for noise budget ops)
    BATCH_GPU_PREFER = 1 << 3,      // Prefer GPU execution if available
    BATCH_CPU_FORCE = 1 << 4,       // Force CPU execution
};

// Batch operation result
struct BatchResult {
    bool success;
    size_t processed;
    size_t failed;
    std::string error;
};

/**
 * @brief Batch bootstrapping - refresh noise for multiple ciphertexts
 * 
 * This is the core primitive for GPU acceleration. Each bootstrap is ~13ms
 * on CPU but can be parallelized across thousands of ciphertexts on GPU.
 * 
 * @param cc BinFHE context
 * @param ct_in Input ciphertexts to bootstrap
 * @param ct_out Output bootstrapped ciphertexts (resized if needed)
 * @param flags Batch processing flags
 * @return BatchResult with success/failure counts
 */
BatchResult BootstrapBatch(
    BinFHEContext& cc,
    const std::vector<LWECiphertext>& ct_in,
    std::vector<LWECiphertext>& ct_out,
    uint32_t flags = BATCH_DEFAULT
);

/**
 * @brief Batch function evaluation via LUT bootstrapping
 * 
 * Evaluates an arbitrary function on multiple ciphertexts using a lookup table.
 * This is the primitive for radix integer arithmetic (add_with_carry, etc.)
 * 
 * @param cc BinFHE context
 * @param ct_in Input ciphertexts
 * @param lut Lookup table defining the function
 * @param ct_out Output ciphertexts
 * @param flags Batch processing flags
 * @return BatchResult
 */
BatchResult EvalFuncBatch(
    BinFHEContext& cc,
    const std::vector<LWECiphertext>& ct_in,
    const std::vector<NativeInteger>& lut,
    std::vector<LWECiphertext>& ct_out,
    uint32_t flags = BATCH_DEFAULT
);

/**
 * @brief Batch function evaluation with multiple outputs per input
 * 
 * For operations like add_limb_with_carry that produce (sum, carry) pairs.
 * 
 * @param cc BinFHE context
 * @param ct_in Input ciphertexts
 * @param luts Multiple LUTs, one per output
 * @param ct_out Output ciphertexts (size = ct_in.size() * luts.size())
 * @param flags Batch processing flags
 * @return BatchResult
 */
BatchResult EvalFuncMultiOutputBatch(
    BinFHEContext& cc,
    const std::vector<LWECiphertext>& ct_in,
    const std::vector<std::vector<NativeInteger>>& luts,
    std::vector<LWECiphertext>& ct_out,
    uint32_t flags = BATCH_DEFAULT
);

/**
 * @brief Batch binary gate evaluation
 * 
 * Evaluates the same gate on pairs of ciphertexts.
 * 
 * @param cc BinFHE context
 * @param gate Gate type (AND, OR, XOR, etc.)
 * @param ct1 First input ciphertexts
 * @param ct2 Second input ciphertexts (same size as ct1)
 * @param ct_out Output ciphertexts
 * @param flags Batch processing flags
 * @return BatchResult
 */
BatchResult EvalBinGateBatch(
    BinFHEContext& cc,
    BINGATE gate,
    const std::vector<LWECiphertext>& ct1,
    const std::vector<LWECiphertext>& ct2,
    std::vector<LWECiphertext>& ct_out,
    uint32_t flags = BATCH_DEFAULT
);

/**
 * @brief Batch CMUX (multiplexer) evaluation
 * 
 * For each i: ct_out[i] = ct_sel[i] ? ct_true[i] : ct_false[i]
 * 
 * @param cc BinFHE context
 * @param ct_sel Selector ciphertexts (encrypted booleans)
 * @param ct_true Values if selector is true
 * @param ct_false Values if selector is false
 * @param ct_out Output ciphertexts
 * @param flags Batch processing flags
 * @return BatchResult
 */
BatchResult EvalCMUXBatch(
    BinFHEContext& cc,
    const std::vector<LWECiphertext>& ct_sel,
    const std::vector<LWECiphertext>& ct_true,
    const std::vector<LWECiphertext>& ct_false,
    std::vector<LWECiphertext>& ct_out,
    uint32_t flags = BATCH_DEFAULT
);

/**
 * @brief Batch key switching
 * 
 * Switch ciphertexts from one key to another.
 * 
 * @param cc BinFHE context
 * @param ct_in Input ciphertexts
 * @param ct_out Output ciphertexts under new key
 * @param flags Batch processing flags
 * @return BatchResult
 */
BatchResult KeySwitchBatch(
    BinFHEContext& cc,
    const std::vector<LWECiphertext>& ct_in,
    std::vector<LWECiphertext>& ct_out,
    uint32_t flags = BATCH_DEFAULT
);

/**
 * @brief Batch modulus switching
 * 
 * Switch ciphertexts to a smaller modulus.
 * 
 * @param cc BinFHE context
 * @param ct_in Input ciphertexts
 * @param ct_out Output ciphertexts with new modulus
 * @param flags Batch processing flags
 * @return BatchResult
 */
BatchResult ModSwitchBatch(
    BinFHEContext& cc,
    const std::vector<LWECiphertext>& ct_in,
    std::vector<LWECiphertext>& ct_out,
    uint32_t flags = BATCH_DEFAULT
);

// ============================================================================
// Async batch operations
// ============================================================================

// Future-like handle for async batch operations
class BatchFuture {
public:
    BatchFuture() = default;
    ~BatchFuture();

    // Block until operation completes
    BatchResult Wait();

    // Check if operation is complete
    bool IsReady() const;

    // Cancel operation (if not yet started)
    bool Cancel();

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;

    // Allow async functions to construct futures
    friend BatchFuture BootstrapBatchAsync(BinFHEContext&, const std::vector<LWECiphertext>&, std::vector<LWECiphertext>&);
    friend BatchFuture EvalFuncBatchAsync(BinFHEContext&, const std::vector<LWECiphertext>&, const std::vector<NativeInteger>&, std::vector<LWECiphertext>&);
};

/**
 * @brief Async batch bootstrapping
 * 
 * Returns immediately with a future. Use Wait() to get results.
 */
BatchFuture BootstrapBatchAsync(
    BinFHEContext& cc,
    const std::vector<LWECiphertext>& ct_in,
    std::vector<LWECiphertext>& ct_out
);

/**
 * @brief Async batch function evaluation
 */
BatchFuture EvalFuncBatchAsync(
    BinFHEContext& cc,
    const std::vector<LWECiphertext>& ct_in,
    const std::vector<NativeInteger>& lut,
    std::vector<LWECiphertext>& ct_out
);

// ============================================================================
// Batch scheduling and DAG execution
// ============================================================================

/**
 * @brief Operation node for DAG scheduling
 */
struct BatchOp {
    enum Type {
        OP_BOOTSTRAP,
        OP_EVAL_FUNC,
        OP_BINGATE,
        OP_CMUX,
        OP_KEYSWITCH,
        OP_MODSWITCH
    };
    
    Type type;
    std::vector<size_t> input_ids;     // IDs of input ciphertexts
    std::vector<size_t> output_ids;    // IDs of output ciphertexts
    std::vector<NativeInteger> lut;    // For EVAL_FUNC
    BINGATE gate;                       // For BINGATE
};

/**
 * @brief Batch operation DAG for optimal scheduling
 * 
 * Collects multiple operations and executes them with optimal batching
 * and minimal bootstrapping.
 */
class BatchDAG {
public:
    BatchDAG(BinFHEContext& cc);
    ~BatchDAG();
    
    // Add operations to the DAG
    size_t AddCiphertext(const LWECiphertext& ct);
    size_t AddBootstrap(size_t input_id);
    size_t AddEvalFunc(size_t input_id, const std::vector<NativeInteger>& lut);
    size_t AddBinGate(BINGATE gate, size_t input1_id, size_t input2_id);
    size_t AddCMUX(size_t sel_id, size_t true_id, size_t false_id);
    
    // Execute the DAG
    BatchResult Execute(uint32_t flags = BATCH_DEFAULT);
    
    // Get output ciphertext
    LWECiphertext GetResult(size_t id) const;
    
    // Clear the DAG for reuse
    void Clear();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace lux::fhe

#endif // BINFHE_BATCH_H
