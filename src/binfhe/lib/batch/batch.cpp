// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// Batch API implementation - routes to backend

#include "batch/binfhe-batch.h"
#include "backend/backend.h"
#include <stdexcept>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace lbcrypto {

// ============================================================================
// BatchFuture Implementation
// ============================================================================

struct BatchFuture::Impl {
    std::future<BatchResult> future;
    std::atomic<bool> cancelled{false};
    std::atomic<bool> ready{false};
    BatchResult result;
};

BatchFuture::~BatchFuture() = default;

BatchResult BatchFuture::Wait() {
    if (impl_ && impl_->future.valid()) {
        impl_->result = impl_->future.get();
        impl_->ready = true;
    }
    return impl_ ? impl_->result : BatchResult{false, 0, 0, "No operation"};
}

bool BatchFuture::IsReady() const {
    return impl_ && impl_->ready;
}

bool BatchFuture::Cancel() {
    if (impl_ && !impl_->ready) {
        impl_->cancelled = true;
        return true;
    }
    return false;
}

// ============================================================================
// Batch Operations
// ============================================================================

BatchResult BootstrapBatch(
    BinFHEContext& cc,
    const std::vector<LWECiphertext>& ct_in,
    std::vector<LWECiphertext>& ct_out,
    uint32_t flags
) {
    BatchResult result{true, 0, 0, ""};
    
    if (ct_in.empty()) {
        return result;
    }
    
    try {
        // Get the appropriate backend
        auto* backend = backend::BackendRegistry::Instance().GetDefault();
        if (!backend) {
            return BatchResult{false, 0, ct_in.size(), "No backend available"};
        }
        
        // Get crypto params from context
        auto params = cc.GetParams()->GetRingGSWParams();
        auto ek = cc.GetRefreshKey();
        
        // Prepare accumulators
        std::vector<RLWECiphertext> accs(ct_in.size());
        
        // Initialize accumulators with LUT (identity for bootstrap)
        // In a full implementation, this would set up the test polynomial
        for (size_t i = 0; i < ct_in.size(); ++i) {
            // accs[i] would be initialized from LUT
        }
        
        // Call backend batch blind rotate
        backend->BlindRotateBatch(params, ct_in, ek, accs);
        
        // Extract LWE from RLWE (sample extraction)
        ct_out.resize(ct_in.size());
        for (size_t i = 0; i < ct_in.size(); ++i) {
            // Extract constant term from RLWE
            // ct_out[i] = cc.EvalFloor(accs[i], ...);
            ct_out[i] = ct_in[i]; // Placeholder
        }
        
        result.processed = ct_in.size();
    } catch (const std::exception& e) {
        result.success = false;
        result.failed = ct_in.size();
        result.error = e.what();
    }
    
    return result;
}

BatchResult EvalFuncBatch(
    BinFHEContext& cc,
    const std::vector<LWECiphertext>& ct_in,
    const std::vector<NativeInteger>& lut,
    std::vector<LWECiphertext>& ct_out,
    uint32_t flags
) {
    BatchResult result{true, 0, 0, ""};
    
    if (ct_in.empty()) {
        return result;
    }
    
    try {
        ct_out.resize(ct_in.size());
        
        // For each ciphertext, evaluate the LUT via bootstrapping
        // This is the core TFHE operation for radix arithmetic
        #pragma omp parallel for if(ct_in.size() > 4 && !(flags & BATCH_ASYNC))
        for (size_t i = 0; i < ct_in.size(); ++i) {
            // Evaluate function by encoding LUT in test polynomial
            // and running blind rotation
            ct_out[i] = cc.EvalFunc(ct_in[i], lut);
        }
        
        result.processed = ct_in.size();
    } catch (const std::exception& e) {
        result.success = false;
        result.failed = ct_in.size();
        result.error = e.what();
    }
    
    return result;
}

BatchResult EvalFuncMultiOutputBatch(
    BinFHEContext& cc,
    const std::vector<LWECiphertext>& ct_in,
    const std::vector<std::vector<NativeInteger>>& luts,
    std::vector<LWECiphertext>& ct_out,
    uint32_t flags
) {
    BatchResult result{true, 0, 0, ""};
    
    if (ct_in.empty() || luts.empty()) {
        return result;
    }
    
    try {
        size_t num_outputs = luts.size();
        ct_out.resize(ct_in.size() * num_outputs);
        
        // For each input, evaluate all LUTs
        #pragma omp parallel for if(ct_in.size() > 4)
        for (size_t i = 0; i < ct_in.size(); ++i) {
            for (size_t j = 0; j < num_outputs; ++j) {
                ct_out[i * num_outputs + j] = cc.EvalFunc(ct_in[i], luts[j]);
            }
        }
        
        result.processed = ct_in.size();
    } catch (const std::exception& e) {
        result.success = false;
        result.failed = ct_in.size();
        result.error = e.what();
    }
    
    return result;
}

BatchResult EvalBinGateBatch(
    BinFHEContext& cc,
    BINGATE gate,
    const std::vector<LWECiphertext>& ct1,
    const std::vector<LWECiphertext>& ct2,
    std::vector<LWECiphertext>& ct_out,
    uint32_t flags
) {
    BatchResult result{true, 0, 0, ""};
    
    if (ct1.size() != ct2.size()) {
        return BatchResult{false, 0, ct1.size(), "Input size mismatch"};
    }
    
    if (ct1.empty()) {
        return result;
    }
    
    try {
        ct_out.resize(ct1.size());
        
        #pragma omp parallel for if(ct1.size() > 4)
        for (size_t i = 0; i < ct1.size(); ++i) {
            ct_out[i] = cc.EvalBinGate(gate, ct1[i], ct2[i]);
        }
        
        result.processed = ct1.size();
    } catch (const std::exception& e) {
        result.success = false;
        result.failed = ct1.size();
        result.error = e.what();
    }
    
    return result;
}

BatchResult EvalCMUXBatch(
    BinFHEContext& cc,
    const std::vector<LWECiphertext>& ct_sel,
    const std::vector<LWECiphertext>& ct_true,
    const std::vector<LWECiphertext>& ct_false,
    std::vector<LWECiphertext>& ct_out,
    uint32_t flags
) {
    BatchResult result{true, 0, 0, ""};
    
    if (ct_sel.size() != ct_true.size() || ct_sel.size() != ct_false.size()) {
        return BatchResult{false, 0, ct_sel.size(), "Input size mismatch"};
    }
    
    if (ct_sel.empty()) {
        return result;
    }
    
    try {
        ct_out.resize(ct_sel.size());
        
        #pragma omp parallel for if(ct_sel.size() > 4)
        for (size_t i = 0; i < ct_sel.size(); ++i) {
            // CMUX = sel ? true : false
            // EvalBinGate for 3-input gates (CMUX, MAJORITY, AND3, OR3) takes a vector
            std::vector<LWECiphertext> inputs = {ct_sel[i], ct_true[i], ct_false[i]};
            ct_out[i] = cc.EvalBinGate(CMUX, inputs);
        }
        
        result.processed = ct_sel.size();
    } catch (const std::exception& e) {
        result.success = false;
        result.failed = ct_sel.size();
        result.error = e.what();
    }
    
    return result;
}

BatchResult KeySwitchBatch(
    BinFHEContext& cc,
    const std::vector<LWECiphertext>& ct_in,
    std::vector<LWECiphertext>& ct_out,
    uint32_t flags
) {
    BatchResult result{true, 0, 0, ""};
    
    if (ct_in.empty()) {
        return result;
    }
    
    try {
        auto* backend = backend::BackendRegistry::Instance().GetDefault();
        if (!backend) {
            return BatchResult{false, 0, ct_in.size(), "No backend available"};
        }
        
        auto params = cc.GetParams()->GetLWEParams();
        auto ks = cc.GetSwitchKey();
        
        backend->KeySwitchBatch(params, ct_in, ks, ct_out);
        
        result.processed = ct_in.size();
    } catch (const std::exception& e) {
        result.success = false;
        result.failed = ct_in.size();
        result.error = e.what();
    }
    
    return result;
}

BatchResult ModSwitchBatch(
    BinFHEContext& cc,
    const std::vector<LWECiphertext>& ct_in,
    std::vector<LWECiphertext>& ct_out,
    uint32_t flags
) {
    BatchResult result{true, 0, 0, ""};
    
    if (ct_in.empty()) {
        return result;
    }
    
    try {
        auto* backend = backend::BackendRegistry::Instance().GetDefault();
        if (!backend) {
            return BatchResult{false, 0, ct_in.size(), "No backend available"};
        }
        
        auto params = cc.GetParams()->GetLWEParams();
        
        backend->ModSwitchBatch(params, ct_in, ct_out);
        
        result.processed = ct_in.size();
    } catch (const std::exception& e) {
        result.success = false;
        result.failed = ct_in.size();
        result.error = e.what();
    }
    
    return result;
}

// ============================================================================
// Async Operations
// ============================================================================

BatchFuture BootstrapBatchAsync(
    BinFHEContext& cc,
    const std::vector<LWECiphertext>& ct_in,
    std::vector<LWECiphertext>& ct_out
) {
    BatchFuture future;
    future.impl_ = std::make_shared<BatchFuture::Impl>();
    
    future.impl_->future = std::async(std::launch::async, [&cc, &ct_in, &ct_out]() {
        return BootstrapBatch(cc, ct_in, ct_out, BATCH_ASYNC);
    });
    
    return future;
}

BatchFuture EvalFuncBatchAsync(
    BinFHEContext& cc,
    const std::vector<LWECiphertext>& ct_in,
    const std::vector<NativeInteger>& lut,
    std::vector<LWECiphertext>& ct_out
) {
    BatchFuture future;
    future.impl_ = std::make_shared<BatchFuture::Impl>();
    
    future.impl_->future = std::async(std::launch::async, [&cc, &ct_in, &lut, &ct_out]() {
        return EvalFuncBatch(cc, ct_in, lut, ct_out, BATCH_ASYNC);
    });
    
    return future;
}

// ============================================================================
// BatchDAG Implementation
// ============================================================================

struct BatchDAG::Impl {
    BinFHEContext& cc;
    std::vector<LWECiphertext> ciphertexts;
    std::vector<BatchOp> operations;
    std::vector<std::vector<size_t>> dependencies;  // Which ops depend on each op
    
    Impl(BinFHEContext& context) : cc(context) {}
};

BatchDAG::BatchDAG(BinFHEContext& cc) 
    : impl_(std::make_unique<Impl>(cc)) {}

BatchDAG::~BatchDAG() = default;

size_t BatchDAG::AddCiphertext(const LWECiphertext& ct) {
    size_t id = impl_->ciphertexts.size();
    impl_->ciphertexts.push_back(ct);
    return id;
}

size_t BatchDAG::AddBootstrap(size_t input_id) {
    size_t output_id = impl_->ciphertexts.size();
    impl_->ciphertexts.push_back(nullptr);  // Placeholder

    BatchOp op;
    op.type = BatchOp::OP_BOOTSTRAP;
    op.input_ids = {input_id};
    op.output_ids = {output_id};
    impl_->operations.push_back(op);

    return output_id;
}

size_t BatchDAG::AddEvalFunc(size_t input_id, const std::vector<NativeInteger>& lut) {
    size_t output_id = impl_->ciphertexts.size();
    impl_->ciphertexts.push_back(nullptr);
    
    BatchOp op;
    op.type = BatchOp::OP_EVAL_FUNC;
    op.input_ids = {input_id};
    op.output_ids = {output_id};
    op.lut = lut;
    impl_->operations.push_back(op);
    
    return output_id;
}

size_t BatchDAG::AddBinGate(BINGATE gate, size_t input1_id, size_t input2_id) {
    size_t output_id = impl_->ciphertexts.size();
    impl_->ciphertexts.push_back(nullptr);
    
    BatchOp op;
    op.type = BatchOp::OP_BINGATE;
    op.input_ids = {input1_id, input2_id};
    op.output_ids = {output_id};
    op.gate = gate;
    impl_->operations.push_back(op);
    
    return output_id;
}

size_t BatchDAG::AddCMUX(size_t sel_id, size_t true_id, size_t false_id) {
    size_t output_id = impl_->ciphertexts.size();
    impl_->ciphertexts.push_back(nullptr);
    
    BatchOp op;
    op.type = BatchOp::OP_CMUX;
    op.input_ids = {sel_id, true_id, false_id};
    op.output_ids = {output_id};
    impl_->operations.push_back(op);
    
    return output_id;
}

BatchResult BatchDAG::Execute(uint32_t flags) {
    BatchResult result{true, 0, 0, ""};
    
    try {
        // Topological sort operations
        // Group by type for batching
        // Execute in optimal order
        
        // Simple sequential execution for now
        for (const auto& op : impl_->operations) {
            switch (op.type) {
                case BatchOp::OP_BOOTSTRAP: {
                    std::vector<LWECiphertext> in = {impl_->ciphertexts[op.input_ids[0]]};
                    std::vector<LWECiphertext> out;
                    BootstrapBatch(impl_->cc, in, out, flags);
                    impl_->ciphertexts[op.output_ids[0]] = out[0];
                    break;
                }
                case BatchOp::OP_EVAL_FUNC: {
                    std::vector<LWECiphertext> in = {impl_->ciphertexts[op.input_ids[0]]};
                    std::vector<LWECiphertext> out;
                    EvalFuncBatch(impl_->cc, in, op.lut, out, flags);
                    impl_->ciphertexts[op.output_ids[0]] = out[0];
                    break;
                }
                case BatchOp::OP_BINGATE: {
                    std::vector<LWECiphertext> ct1 = {impl_->ciphertexts[op.input_ids[0]]};
                    std::vector<LWECiphertext> ct2 = {impl_->ciphertexts[op.input_ids[1]]};
                    std::vector<LWECiphertext> out;
                    EvalBinGateBatch(impl_->cc, op.gate, ct1, ct2, out, flags);
                    impl_->ciphertexts[op.output_ids[0]] = out[0];
                    break;
                }
                case BatchOp::OP_CMUX: {
                    std::vector<LWECiphertext> sel = {impl_->ciphertexts[op.input_ids[0]]};
                    std::vector<LWECiphertext> t = {impl_->ciphertexts[op.input_ids[1]]};
                    std::vector<LWECiphertext> f = {impl_->ciphertexts[op.input_ids[2]]};
                    std::vector<LWECiphertext> out;
                    EvalCMUXBatch(impl_->cc, sel, t, f, out, flags);
                    impl_->ciphertexts[op.output_ids[0]] = out[0];
                    break;
                }
                default:
                    break;
            }
            result.processed++;
        }
    } catch (const std::exception& e) {
        result.success = false;
        result.error = e.what();
    }
    
    return result;
}

LWECiphertext BatchDAG::GetResult(size_t id) const {
    if (id >= impl_->ciphertexts.size()) {
        throw std::out_of_range("Invalid ciphertext ID");
    }
    return impl_->ciphertexts[id];
}

void BatchDAG::Clear() {
    impl_->ciphertexts.clear();
    impl_->operations.clear();
    impl_->dependencies.clear();
}

} // namespace lbcrypto
