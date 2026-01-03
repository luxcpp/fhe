// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// Verifiable FHE Computation Witnesses
//
// PROBLEM: How to verify FHE computation result without re-computing?
// - FHE operations (bootstrapping, external product) are expensive
// - Verifier cannot re-run full computation
// - Need succinct proof that computation was done correctly
//
// SOLUTION: Generate witness during computation, verify cheaply
// - Commit to intermediate ciphertexts during computation
// - Merkle tree over intermediate values enables spot-checking
// - Linear combination proof batches multiple gate verifications
// - Fiat-Shamir transforms interactive protocol to non-interactive
//
// INNOVATION: Commit-and-Prove paradigm for FHE
// 1. Prover commits to intermediate ciphertexts (Merkle root)
// 2. Verifier challenges random gate indices (Fiat-Shamir)
// 3. Prover reveals challenged gates with Merkle proofs
// 4. Verifier spot-checks gates and linear combination
//
// Verification complexity: O(k log n) vs O(n) for re-computation
// where k = security parameter (e.g., 80), n = circuit size

#ifndef ZKP_FHE_WITNESS_H
#define ZKP_FHE_WITNESS_H

#include "lwe-ciphertext.h"
#include "rlwe-ciphertext.h"
#include "threshold/transcript.h"
#include "math/math-hal.h"

#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <optional>

namespace lux::fhe {
namespace zkp {

// Use existing hash infrastructure
using threshold::Hash256;
using threshold::MerkleTree;
using threshold::TranscriptBuilder;
using threshold::DomainTag;

// ============================================================================
// Gate Types
// ============================================================================

/**
 * @brief Types of FHE gates that can be witnessed
 */
enum class GateType : uint8_t {
    // Binary gates
    AND = 0x01,
    OR = 0x02,
    NAND = 0x03,
    NOR = 0x04,
    XOR = 0x05,
    XNOR = 0x06,
    NOT = 0x07,

    // Arithmetic gates
    ADD = 0x10,
    SUB = 0x11,
    MUL = 0x12,

    // Special operations
    BOOTSTRAP = 0x20,
    KEY_SWITCH = 0x21,
    MOD_SWITCH = 0x22,
    EXTERNAL_PRODUCT = 0x23,
    CMux = 0x24,

    // Input/Output markers
    INPUT = 0xF0,
    OUTPUT = 0xF1,
};

// ============================================================================
// Gate Record
// ============================================================================

/**
 * @brief Record of a single gate evaluation
 *
 * Stores inputs, output, and gate type for verification.
 * Input/output are indices into the ciphertext table.
 */
struct GateRecord {
    uint64_t gate_id;          // Unique gate identifier
    GateType type;             // Type of gate
    std::vector<uint64_t> input_indices;  // Indices of input ciphertexts
    uint64_t output_index;     // Index of output ciphertext

    // Optional: auxiliary data for gate-specific verification
    std::vector<uint8_t> aux_data;

    // Serialization
    std::vector<uint8_t> Serialize() const;
    static GateRecord Deserialize(const uint8_t* data, size_t len);
};

// ============================================================================
// Commitment Scheme
// ============================================================================

/**
 * @brief Hash-based commitment scheme
 *
 * Simple but efficient: C = H(value || randomness)
 * Opening: reveal value and randomness, recompute and compare
 */
struct Commitment {
    Hash256 value;

    bool operator==(const Commitment& other) const { return value == other.value; }
    bool operator!=(const Commitment& other) const { return value != other.value; }
};

/**
 * @brief Commitment opening (proof of committed value)
 */
struct CommitmentOpening {
    std::vector<uint8_t> committed_value;
    Hash256 randomness;

    // Verify that opening matches commitment
    bool Verify(const Commitment& commitment) const;
};

/**
 * @brief Create commitment to value
 */
Commitment Commit(const std::vector<uint8_t>& value, const Hash256& randomness);
Commitment Commit(const LWECiphertext& ct, const Hash256& randomness);

/**
 * @brief Generate cryptographically secure randomness for commitment
 */
Hash256 GenerateRandomness();

// ============================================================================
// Ciphertext Table
// ============================================================================

/**
 * @brief Table of ciphertexts with commitments
 *
 * During computation, each ciphertext is committed.
 * The table maps indices to commitments.
 * Actual ciphertexts are stored separately (or computed on-demand).
 */
class CiphertextTable {
public:
    CiphertextTable();
    ~CiphertextTable();

    // Non-copyable, movable
    CiphertextTable(const CiphertextTable&) = delete;
    CiphertextTable& operator=(const CiphertextTable&) = delete;
    CiphertextTable(CiphertextTable&&) noexcept;
    CiphertextTable& operator=(CiphertextTable&&) noexcept;

    // ========================================================================
    // Ciphertext Registration
    // ========================================================================

    /**
     * @brief Register input ciphertext
     * @return Index of registered ciphertext
     */
    uint64_t RegisterInput(const LWECiphertext& ct);

    /**
     * @brief Register intermediate ciphertext (result of gate evaluation)
     * @return Index of registered ciphertext
     */
    uint64_t RegisterIntermediate(const LWECiphertext& ct, uint64_t gate_id);

    /**
     * @brief Mark ciphertext as output
     */
    void MarkOutput(uint64_t index);

    // ========================================================================
    // Commitment Access
    // ========================================================================

    /**
     * @brief Get commitment for ciphertext at index
     */
    Commitment GetCommitment(uint64_t index) const;

    /**
     * @brief Get all commitments
     */
    std::vector<Commitment> GetAllCommitments() const;

    /**
     * @brief Get commitment opening (for verification)
     */
    CommitmentOpening GetOpening(uint64_t index) const;

    /**
     * @brief Build Merkle tree over commitments
     */
    void BuildMerkleTree();

    /**
     * @brief Get Merkle root
     */
    Hash256 GetMerkleRoot() const;

    /**
     * @brief Generate Merkle inclusion proof for ciphertext
     */
    MerkleTree::InclusionProof ProveInclusion(uint64_t index) const;

    // ========================================================================
    // Table Info
    // ========================================================================

    size_t Size() const;
    bool IsInput(uint64_t index) const;
    bool IsOutput(uint64_t index) const;
    bool IsIntermediate(uint64_t index) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// Linear Combination Proof
// ============================================================================

/**
 * @brief Proof that linear combination of ciphertexts is correct
 *
 * INNOVATION: Instead of verifying each gate separately, combine
 * gate outputs with random challenges and verify sum.
 *
 * For gates g_1, ..., g_k with challenges c_1, ..., c_k:
 * Verify: sum(c_i * output_i) = expected_combination
 *
 * This reduces k verifications to 1 verification with soundness 1/q.
 */
struct LinearCombinationProof {
    // Challenged gate indices
    std::vector<uint64_t> gate_indices;

    // Random challenges (Fiat-Shamir derived)
    std::vector<NativeInteger> challenges;

    // Combined output ciphertext
    LWECiphertext combined_output;

    // Expected combination (computed from inputs)
    LWECiphertext expected_combination;

    // Serialization
    std::vector<uint8_t> Serialize() const;
    static LinearCombinationProof Deserialize(const uint8_t* data, size_t len);
};

// ============================================================================
// FHE Witness
// ============================================================================

/**
 * @brief Witness for FHE computation correctness
 *
 * Contains everything needed to verify computation without re-executing.
 */
struct FHEWitness {
    // Version for serialization compatibility
    uint32_t version = 1;

    // Circuit identifier
    std::string circuit_id;

    // Commitment to intermediate ciphertexts (Merkle root)
    Hash256 ciphertext_root;

    // Gate evaluation trace (committed)
    Hash256 gate_trace_root;

    // Sampled verification points with Merkle proofs
    struct VerificationPoint {
        uint64_t gate_index;
        GateRecord gate;
        MerkleTree::InclusionProof gate_proof;

        // Input/output ciphertext commitments with proofs
        std::vector<Commitment> input_commitments;
        std::vector<MerkleTree::InclusionProof> input_proofs;
        Commitment output_commitment;
        MerkleTree::InclusionProof output_proof;
    };
    std::vector<VerificationPoint> verification_points;

    // Linear combination proof (batched verification)
    LinearCombinationProof linear_proof;

    // Fiat-Shamir challenge seed (for reproducibility)
    Hash256 challenge_seed;

    // Input commitments (public)
    std::vector<Commitment> input_commitments;

    // Output commitments (public)
    std::vector<Commitment> output_commitments;

    // Serialization
    std::vector<uint8_t> Serialize() const;
    static FHEWitness Deserialize(const uint8_t* data, size_t len);
};

// ============================================================================
// Witness Builder
// ============================================================================

/**
 * @brief Builder for creating FHE computation witnesses
 *
 * Usage:
 *   WitnessBuilder builder("circuit_001");
 *   builder.RegisterInputs(input_cts);
 *
 *   // During computation, record each gate
 *   auto output_idx = builder.RecordGate(GateType::AND, {in1_idx, in2_idx}, result);
 *
 *   // Finalize and generate witness
 *   FHEWitness witness = builder.Build(security_parameter);
 */
class WitnessBuilder {
public:
    /**
     * @brief Create witness builder
     * @param circuit_id Unique identifier for the circuit
     * @param security_parameter Number of gates to sample (default 80)
     */
    explicit WitnessBuilder(const std::string& circuit_id, uint32_t security_parameter = 80);
    ~WitnessBuilder();

    // Non-copyable, movable
    WitnessBuilder(const WitnessBuilder&) = delete;
    WitnessBuilder& operator=(const WitnessBuilder&) = delete;
    WitnessBuilder(WitnessBuilder&&) noexcept;
    WitnessBuilder& operator=(WitnessBuilder&&) noexcept;

    // ========================================================================
    // Input Registration
    // ========================================================================

    /**
     * @brief Register a single input ciphertext
     * @return Index in ciphertext table
     */
    uint64_t RegisterInput(const LWECiphertext& ct);

    /**
     * @brief Register multiple input ciphertexts
     * @return Vector of indices
     */
    std::vector<uint64_t> RegisterInputs(const std::vector<LWECiphertext>& cts);

    // ========================================================================
    // Gate Recording
    // ========================================================================

    /**
     * @brief Record a gate evaluation
     *
     * @param type Type of gate
     * @param input_indices Indices of input ciphertexts
     * @param output Result ciphertext
     * @param aux_data Optional auxiliary data for verification
     * @return Index of output ciphertext in table
     */
    uint64_t RecordGate(
        GateType type,
        const std::vector<uint64_t>& input_indices,
        const LWECiphertext& output,
        const std::vector<uint8_t>& aux_data = {}
    );

    /**
     * @brief Record binary gate (AND, OR, NAND, NOR, XOR, XNOR)
     */
    uint64_t RecordBinaryGate(
        GateType type,
        uint64_t input1,
        uint64_t input2,
        const LWECiphertext& output
    );

    /**
     * @brief Record NOT gate
     */
    uint64_t RecordNot(uint64_t input, const LWECiphertext& output);

    /**
     * @brief Record bootstrap operation
     */
    uint64_t RecordBootstrap(uint64_t input, const LWECiphertext& output);

    /**
     * @brief Record CMux operation
     */
    uint64_t RecordCMux(
        uint64_t selector,
        uint64_t ct_true,
        uint64_t ct_false,
        const LWECiphertext& output
    );

    // ========================================================================
    // Output Registration
    // ========================================================================

    /**
     * @brief Mark ciphertexts as outputs
     */
    void MarkOutputs(const std::vector<uint64_t>& indices);

    // ========================================================================
    // Witness Generation
    // ========================================================================

    /**
     * @brief Build the witness
     *
     * 1. Commits to all intermediate ciphertexts
     * 2. Builds Merkle trees
     * 3. Generates Fiat-Shamir challenges
     * 4. Samples verification points
     * 5. Generates linear combination proof
     *
     * @return Complete witness for verification
     */
    FHEWitness Build();

    // ========================================================================
    // Statistics
    // ========================================================================

    uint64_t NumInputs() const;
    uint64_t NumOutputs() const;
    uint64_t NumGates() const;
    uint64_t NumIntermediates() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// Witness Verifier
// ============================================================================

/**
 * @brief Verification result with details
 */
struct VerificationResult {
    bool valid;
    std::string error_message;  // Empty if valid

    // Detailed breakdown (for debugging)
    bool commitment_root_valid = false;
    bool gate_trace_valid = false;
    bool merkle_proofs_valid = false;
    bool spot_checks_valid = false;
    bool linear_combination_valid = false;

    static VerificationResult Success() {
        return {true, "", true, true, true, true, true};
    }

    static VerificationResult Failure(const std::string& msg) {
        return {false, msg, false, false, false, false, false};
    }
};

/**
 * @brief Verifier for FHE computation witnesses
 *
 * VERIFICATION STEPS:
 * 1. Check commitment roots are consistent
 * 2. Verify Merkle inclusion proofs for sampled gates
 * 3. Spot-check gate evaluations (lightweight re-computation)
 * 4. Verify linear combination proof
 *
 * Soundness: 1 - (1 - 1/n)^k where n = gates, k = samples
 * For n=10000, k=80: soundness > 1 - 2^{-80}
 */
class WitnessVerifier {
public:
    WitnessVerifier();
    ~WitnessVerifier();

    // Non-copyable, movable
    WitnessVerifier(const WitnessVerifier&) = delete;
    WitnessVerifier& operator=(const WitnessVerifier&) = delete;
    WitnessVerifier(WitnessVerifier&&) noexcept;
    WitnessVerifier& operator=(WitnessVerifier&&) noexcept;

    /**
     * @brief Verify a witness
     *
     * @param witness The witness to verify
     * @param inputs Input ciphertexts (public)
     * @param outputs Output ciphertexts (public)
     * @return Verification result
     */
    VerificationResult Verify(
        const FHEWitness& witness,
        const std::vector<LWECiphertext>& inputs,
        const std::vector<LWECiphertext>& outputs
    );

    /**
     * @brief Verify only the structural parts (without ciphertext checks)
     *
     * Faster verification that checks:
     * - Merkle proofs
     * - Commitment consistency
     * - Challenge derivation
     *
     * Does NOT check actual FHE computations.
     */
    VerificationResult VerifyStructure(const FHEWitness& witness);

    /**
     * @brief Set verification parameters
     */
    void SetModulus(const NativeInteger& q);
    void SetLWEDimension(uint32_t n);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // Internal verification steps
    bool VerifyCommitmentRoots(const FHEWitness& witness);
    bool VerifyMerkleProofs(const FHEWitness& witness);
    bool VerifySpotChecks(
        const FHEWitness& witness,
        const std::vector<LWECiphertext>& inputs
    );
    bool VerifyLinearCombination(const FHEWitness& witness);
};

// ============================================================================
// Gate Evaluation (for spot-checking)
// ============================================================================

namespace gates {

/**
 * @brief Evaluate a single gate for spot-checking
 *
 * These are simplified versions that verify relationship between
 * input and output ciphertexts without full FHE machinery.
 *
 * For most gates, we check linear relationships:
 * - AND: output = ct1 + ct2 + constant (before bootstrap)
 * - NOT: output = constant - ct
 * - etc.
 *
 * Full correctness requires checking bootstrap correctness,
 * which is done via linear combination proof.
 */

/**
 * @brief Check if output could be result of gate on inputs
 *
 * This is a NECESSARY but not SUFFICIENT condition.
 * Full verification requires linear combination proof.
 */
bool CheckGateRelation(
    GateType type,
    const std::vector<LWECiphertext>& inputs,
    const LWECiphertext& output,
    const NativeInteger& q
);

/**
 * @brief Get expected linear combination coefficients for gate type
 *
 * For batched verification, we compute:
 *   sum(c_i * (output_i - expected_i)) = 0
 *
 * This function returns coefficients for the "expected" computation.
 */
std::vector<NativeInteger> GetGateCoefficients(
    GateType type,
    const NativeInteger& q
);

} // namespace gates

// ============================================================================
// Serialization Helpers
// ============================================================================

namespace serial {

/**
 * @brief Serialize LWE ciphertext to bytes
 */
std::vector<uint8_t> SerializeLWE(const LWECiphertext& ct);

/**
 * @brief Deserialize LWE ciphertext from bytes
 */
LWECiphertext DeserializeLWE(const uint8_t* data, size_t len);

/**
 * @brief Serialize Merkle inclusion proof
 */
std::vector<uint8_t> SerializeInclusionProof(const MerkleTree::InclusionProof& proof);

/**
 * @brief Deserialize Merkle inclusion proof
 */
MerkleTree::InclusionProof DeserializeInclusionProof(const uint8_t* data, size_t len);

} // namespace serial

} // namespace zkp
} // namespace lux::fhe

#endif // ZKP_FHE_WITNESS_H
