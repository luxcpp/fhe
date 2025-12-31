// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// Batched Threshold FHE Operations
//
// INNOVATION: Threshold FHE adds communication rounds (shares, partial decryptions,
// proofs/commitments). We optimize for blockchain throughput by:
//
// 1. BATCH PARTIAL DECRYPTIONS: Process many ciphertexts per party in one call
//    - Amortizes key loading and NTT setup
//    - Enables GPU parallelism across all ciphertexts
//
// 2. BATCH NTT DISPATCH: All shares * all limbs in one kernel launch
//    - Traditional: for ct in cts: for limb in limbs: NTT(limb)  // n*k launches
//    - Batched: NTT_batch(cts, limbs)  // 1 launch, n*k parallel
//
// 3. AMORTIZED TRANSCRIPT HASHING: Hash once per batch, not per ciphertext
//    - Merkle tree for batch commitment (parallelizable)
//    - Single hash for batch challenge
//    - Derive per-element challenges deterministically
//
// This matters in blockchain settings where throughput >> single-op latency.

#ifndef THRESHOLD_BATCH_THRESHOLD_H
#define THRESHOLD_BATCH_THRESHOLD_H

#include "binfhecontext.h"
#include "lwe-ciphertext.h"
#include "threshold/transcript.h"
#include "batch/binfhe-batch.h"
#include <cstdint>
#include <memory>
#include <vector>
#include <functional>
#include <optional>

namespace lbcrypto {
namespace threshold {

// ============================================================================
// Threshold Configuration
// ============================================================================

/**
 * @brief Threshold scheme parameters
 */
struct ThresholdConfig {
    uint32_t threshold;           // t: minimum parties needed to decrypt
    uint32_t total_parties;       // n: total number of parties
    uint32_t party_id;            // This party's ID (1-indexed)
    bool verify_proofs;           // Whether to verify correctness proofs
    bool generate_proofs;         // Whether to generate correctness proofs

    // Defaults: 2-of-3 threshold
    ThresholdConfig()
        : threshold(2), total_parties(3), party_id(1),
          verify_proofs(true), generate_proofs(true) {}
};

// ============================================================================
// Key Shares
// ============================================================================

/**
 * @brief Secret key share for party i
 *
 * In Shamir's secret sharing over Z_q:
 *   sk = sum_{i in S} lambda_i * sk_i  (for any t-size subset S)
 *
 * where lambda_i are Lagrange coefficients.
 */
struct KeyShare {
    uint32_t party_id;
    NativeVector share;           // sk_i
    Hash256 commitment;           // Feldman commitment to share

    // Serialization
    std::vector<uint8_t> Serialize() const;
    static KeyShare Deserialize(const std::vector<uint8_t>& data);
};

/**
 * @brief Public verification key for party i
 *
 * Used to verify partial decryptions without knowing sk_i.
 */
struct VerificationKey {
    uint32_t party_id;
    NativeVector public_share;    // g^{sk_i} for DLEQ proofs
    std::vector<Hash256> commitments;  // Feldman polynomial commitments

    std::vector<uint8_t> Serialize() const;
    static VerificationKey Deserialize(const std::vector<uint8_t>& data);
};

// ============================================================================
// Partial Decryption
// ============================================================================

/**
 * @brief Partial decryption from party i
 *
 * d_i = <a, sk_i> mod q
 *
 * The full decryption is:
 *   m = b - sum_{i in S} lambda_i * d_i  (for threshold subset S)
 */
struct PartialDecryption {
    uint32_t party_id;
    uint32_t ciphertext_index;    // Index in the batch
    NativeInteger value;          // d_i = <a, sk_i>

    std::vector<uint8_t> Serialize() const;
    static PartialDecryption Deserialize(const std::vector<uint8_t>& data);
};

/**
 * @brief Batched partial decryptions from party i
 *
 * Contains partial decryptions for all ciphertexts in a batch.
 */
struct BatchPartialDecryption {
    uint32_t party_id;
    std::vector<NativeInteger> values;  // d_i for each ciphertext
    Hash256 batch_commitment;           // Merkle root of values

    std::vector<uint8_t> Serialize() const;
    static BatchPartialDecryption Deserialize(const std::vector<uint8_t>& data);
};

// ============================================================================
// Correctness Proofs
// ============================================================================

/**
 * @brief DLEQ proof of correct partial decryption
 *
 * Proves: d_i = <a, sk_i> without revealing sk_i
 *
 * The proof shows discrete log equality:
 *   log_g(vk_i) = log_a(d_i)  (both are sk_i)
 *
 * Uses Chaum-Pedersen protocol in batch form.
 */
struct CorrectnessProof {
    uint32_t party_id;
    Hash256 challenge;            // Fiat-Shamir challenge
    NativeVector response;        // z = r + c * sk_i

    std::vector<uint8_t> Serialize() const;
    static CorrectnessProof Deserialize(const std::vector<uint8_t>& data);
};

/**
 * @brief Batched correctness proofs
 *
 * INNOVATION: Instead of n individual DLEQ proofs, we use a batch proof:
 *
 * 1. Compute commitments R_i = g^{r_i}, A_i = a_i^{r_i} for random r_i
 * 2. Build Merkle tree over (R_i, A_i) pairs
 * 3. Single Fiat-Shamir challenge c from Merkle root
 * 4. Per-element challenges c_i derived from c and index
 * 5. Responses z_i = r_i + c_i * sk
 *
 * Verification:
 *   g^{z_i} == R_i * vk^{c_i}
 *   a_i^{z_i} == A_i * d_i^{c_i}
 */
struct BatchCorrectnessProof {
    uint32_t party_id;
    std::vector<NativeVector> commitments_R;  // g^{r_i} for each ct
    std::vector<NativeInteger> commitments_A; // a_i^{r_i} for each ct
    Hash256 merkle_root;                       // Root of commitment tree
    Hash256 batch_challenge;                   // Single FS challenge
    std::vector<NativeInteger> responses;      // z_i for each ct

    std::vector<uint8_t> Serialize() const;
    static BatchCorrectnessProof Deserialize(const std::vector<uint8_t>& data);
};

// ============================================================================
// Batch Operations Result
// ============================================================================

/**
 * @brief Result of batch threshold operation
 */
struct ThresholdBatchResult {
    bool success;
    size_t processed;
    size_t failed;
    std::string error;

    // Timing breakdown (nanoseconds)
    uint64_t time_compute_ns;     // Partial decrypt computation
    uint64_t time_proof_ns;       // Proof generation/verification
    uint64_t time_hash_ns;        // Transcript hashing
    uint64_t time_total_ns;       // Total time

    static ThresholdBatchResult Success(size_t count) {
        return ThresholdBatchResult{true, count, 0, "", 0, 0, 0, 0};
    }

    static ThresholdBatchResult Failure(const std::string& err, size_t count = 0) {
        return ThresholdBatchResult{false, 0, count, err, 0, 0, 0, 0};
    }
};

// ============================================================================
// Batch Threshold Operations
// ============================================================================

/**
 * @brief Batch partial decryption
 *
 * Computes d_i = <a, sk_i> for all ciphertexts in the batch.
 * Optionally generates correctness proofs.
 *
 * OPTIMIZATION:
 * - Vectorized inner product using NTT representation
 * - Single key share loading (amortized across batch)
 * - GPU dispatch of all ciphertexts in parallel
 *
 * @param cc BinFHE context
 * @param config Threshold configuration
 * @param cts Input ciphertexts
 * @param key_share This party's key share
 * @param out Output partial decryptions
 * @param proof Optional output correctness proof
 * @return Result with timing information
 */
ThresholdBatchResult BatchPartialDecrypt(
    BinFHEContext& cc,
    const ThresholdConfig& config,
    const std::vector<LWECiphertext>& cts,
    const KeyShare& key_share,
    BatchPartialDecryption& out,
    std::optional<BatchCorrectnessProof>* proof = nullptr
);

/**
 * @brief Batch combine shares
 *
 * Combines partial decryptions from t parties to recover plaintexts.
 *
 * m_j = b_j - sum_{i in S} lambda_i * d_{i,j}
 *
 * @param cc BinFHE context
 * @param config Threshold configuration
 * @param cts Input ciphertexts
 * @param partials Partial decryptions from t parties
 * @param plaintexts Output plaintexts
 * @return Result with timing information
 */
ThresholdBatchResult BatchCombineShares(
    BinFHEContext& cc,
    const ThresholdConfig& config,
    const std::vector<LWECiphertext>& cts,
    const std::vector<BatchPartialDecryption>& partials,
    std::vector<LWEPlaintext>& plaintexts
);

/**
 * @brief Batch verify correctness proofs
 *
 * Verifies that partial decryptions are correct without seeing sk_i.
 *
 * OPTIMIZATION:
 * - Batch verification using random linear combination
 * - Single multi-exponentiation instead of n exponentiations
 * - Parallel challenge derivation
 *
 * @param cc BinFHE context
 * @param config Threshold configuration
 * @param cts Ciphertexts that were partially decrypted
 * @param partial Partial decryptions to verify
 * @param proof Correctness proof
 * @param vk Verification key for the party
 * @return Result indicating success/failure
 */
ThresholdBatchResult BatchVerifyProofs(
    BinFHEContext& cc,
    const ThresholdConfig& config,
    const std::vector<LWECiphertext>& cts,
    const BatchPartialDecryption& partial,
    const BatchCorrectnessProof& proof,
    const VerificationKey& vk
);

/**
 * @brief Batch transcript hash
 *
 * Computes a single hash commitment for an entire batch of:
 * - Ciphertexts
 * - Partial decryptions
 * - Correctness proofs
 *
 * @param cts Input ciphertexts
 * @param partials All partial decryptions
 * @param proofs All correctness proofs
 * @return Batch commitment hash
 */
Hash256 BatchTranscriptHash(
    const std::vector<LWECiphertext>& cts,
    const std::vector<BatchPartialDecryption>& partials,
    const std::vector<BatchCorrectnessProof>& proofs
);

// ============================================================================
// Key Generation
// ============================================================================

/**
 * @brief Generate threshold key shares using Shamir's scheme
 *
 * Generates (t, n) threshold shares of a secret key.
 * Each party gets their share securely (not broadcast).
 *
 * @param cc BinFHE context
 * @param config Threshold configuration
 * @param master_key Master secret key (dealer has this)
 * @param shares Output key shares for each party
 * @param vks Output verification keys
 */
void GenerateKeyShares(
    BinFHEContext& cc,
    const ThresholdConfig& config,
    const LWEPrivateKey& master_key,
    std::vector<KeyShare>& shares,
    std::vector<VerificationKey>& vks
);

/**
 * @brief Compute Lagrange coefficients for a party subset
 *
 * lambda_i = prod_{j in S, j != i} (j / (j - i))  mod q
 *
 * @param party_ids IDs of parties in the subset (1-indexed)
 * @param q Modulus
 * @return Lagrange coefficients for each party
 */
std::vector<NativeInteger> ComputeLagrangeCoefficients(
    const std::vector<uint32_t>& party_ids,
    const NativeInteger& q
);

// ============================================================================
// Threshold Decrypt Pipeline
// ============================================================================

/**
 * @brief Complete threshold decryption pipeline
 *
 * Orchestrates the full threshold decryption:
 * 1. Each party computes partial decryptions
 * 2. Parties exchange partial decryptions
 * 3. Verify correctness proofs (optional)
 * 4. Combine shares to recover plaintexts
 *
 * This class handles the protocol flow for a single party.
 */
class ThresholdDecryptPipeline {
public:
    /**
     * @brief Create pipeline for a party
     */
    ThresholdDecryptPipeline(
        BinFHEContext& cc,
        const ThresholdConfig& config,
        const KeyShare& key_share,
        const std::vector<VerificationKey>& all_vks
    );
    ~ThresholdDecryptPipeline();

    // Non-copyable
    ThresholdDecryptPipeline(const ThresholdDecryptPipeline&) = delete;
    ThresholdDecryptPipeline& operator=(const ThresholdDecryptPipeline&) = delete;

    // ========================================================================
    // Pipeline Steps
    // ========================================================================

    /**
     * @brief Step 1: Compute our partial decryptions
     *
     * @param cts Ciphertexts to decrypt
     * @return Our partial decryptions and proof
     */
    std::pair<BatchPartialDecryption, BatchCorrectnessProof>
    ComputePartials(const std::vector<LWECiphertext>& cts);

    /**
     * @brief Step 2: Receive partial decryptions from other parties
     *
     * @param party_id ID of the sending party
     * @param partial Their partial decryptions
     * @param proof Their correctness proof
     * @return True if proof verifies
     */
    bool ReceivePartials(
        uint32_t party_id,
        const BatchPartialDecryption& partial,
        const BatchCorrectnessProof& proof
    );

    /**
     * @brief Step 3: Combine and decrypt
     *
     * Called after receiving t-1 other partials (plus our own).
     *
     * @param plaintexts Output plaintexts
     * @return True if successful
     */
    bool Combine(std::vector<LWEPlaintext>& plaintexts);

    // ========================================================================
    // State Queries
    // ========================================================================

    /**
     * @brief Number of valid partial decryptions received
     */
    size_t NumPartialsReceived() const;

    /**
     * @brief Check if we have enough partials to combine
     */
    bool CanCombine() const;

    /**
     * @brief Get IDs of parties whose partials we have
     */
    std::vector<uint32_t> ReceivedPartyIds() const;

    /**
     * @brief Reset pipeline for new batch
     */
    void Reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// Batched Multi-Party Computation Helpers
// ============================================================================

/**
 * @brief Batch inner product <a, s> for many ciphertexts
 *
 * Computes <a_i, s> for all ciphertexts in parallel.
 *
 * OPTIMIZATION:
 * - If s is in NTT form, keep it there
 * - Batch all NTT conversions for a vectors
 * - Single vectorized multiply-accumulate
 *
 * @param a_vectors The 'a' components of LWE ciphertexts
 * @param s Secret key (or share)
 * @param q Modulus
 * @return Inner products for each ciphertext
 */
std::vector<NativeInteger> BatchInnerProduct(
    const std::vector<NativeVector>& a_vectors,
    const NativeVector& s,
    const NativeInteger& q
);

/**
 * @brief Batch modular operations
 *
 * Computes b_i - sum_j(lambda_j * d_{j,i}) mod q for all i.
 *
 * @param b_values The 'b' components of LWE ciphertexts
 * @param partials Partial decryptions from each party
 * @param lambdas Lagrange coefficients
 * @param q Modulus
 * @return Decrypted values (before rounding)
 */
std::vector<NativeInteger> BatchCombinePartials(
    const std::vector<NativeInteger>& b_values,
    const std::vector<std::vector<NativeInteger>>& partials,
    const std::vector<NativeInteger>& lambdas,
    const NativeInteger& q
);

} // namespace threshold
} // namespace lbcrypto

#endif // THRESHOLD_BATCH_THRESHOLD_H
