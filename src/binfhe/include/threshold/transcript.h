// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// Fiat-Shamir Transcript for Batched Threshold FHE Proofs
//
// INNOVATION: Instead of hashing per-ciphertext (O(n) hash ops), we:
// 1. Build a Merkle tree over batch elements (O(n) hashes, parallelizable)
// 2. Hash the root once for batch challenge
// 3. Derive per-element challenges deterministically from batch challenge
//
// This amortizes transcript overhead from O(n) serial hashes to O(1) + parallel tree.

#ifndef THRESHOLD_TRANSCRIPT_H
#define THRESHOLD_TRANSCRIPT_H

#include "lwe-ciphertext.h"
#include "math/math-hal.h"
#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <array>

namespace lux::fhe {
namespace threshold {

// ============================================================================
// Hash Output Type (256-bit)
// ============================================================================

using Hash256 = std::array<uint8_t, 32>;

// Hash comparison and conversion
bool operator==(const Hash256& a, const Hash256& b);
bool operator!=(const Hash256& a, const Hash256& b);
std::string HashToHex(const Hash256& h);
Hash256 HexToHash(const std::string& hex);

// ============================================================================
// Domain Separation Tags
// ============================================================================

/**
 * @brief Domain separation for different proof types
 *
 * Each proof type uses a unique domain separator to prevent
 * cross-protocol attacks. Tags are prefixed to hash inputs.
 */
enum class DomainTag : uint8_t {
    PARTIAL_DECRYPT = 0x01,      // Threshold partial decryption
    KEY_SHARE = 0x02,            // Key share commitment
    PROOF_CORRECTNESS = 0x03,    // DLEQ proof of correct decryption
    PROOF_RANGE = 0x04,          // Range proof
    COMMITMENT = 0x05,           // Generic commitment
    CHALLENGE = 0x06,            // Challenge derivation
    MERKLE_LEAF = 0x10,          // Merkle tree leaf
    MERKLE_INTERNAL = 0x11,      // Merkle tree internal node
    BATCH_ROOT = 0x20,           // Batch root hash
    ELEMENT_CHALLENGE = 0x21,    // Per-element challenge from batch
};

// ============================================================================
// Transcript Builder
// ============================================================================

/**
 * @brief Fiat-Shamir transcript for non-interactive proofs
 *
 * Accumulates protocol messages and produces challenges. Uses SHAKE256
 * for variable-length output and domain separation.
 *
 * Usage:
 *   TranscriptBuilder tx("ThresholdDecrypt");
 *   tx.Append(DomainTag::PARTIAL_DECRYPT, ciphertext_bytes);
 *   tx.Append(DomainTag::KEY_SHARE, key_share_bytes);
 *   Hash256 challenge = tx.Challenge();
 */
class TranscriptBuilder {
public:
    /**
     * @brief Create transcript with protocol label
     * @param protocol_label Unique identifier for the protocol
     */
    explicit TranscriptBuilder(const std::string& protocol_label);
    ~TranscriptBuilder();

    // Non-copyable, movable
    TranscriptBuilder(const TranscriptBuilder&) = delete;
    TranscriptBuilder& operator=(const TranscriptBuilder&) = delete;
    TranscriptBuilder(TranscriptBuilder&&) noexcept;
    TranscriptBuilder& operator=(TranscriptBuilder&&) noexcept;

    // ========================================================================
    // Append Operations
    // ========================================================================

    /**
     * @brief Append raw bytes with domain tag
     */
    void Append(DomainTag tag, const uint8_t* data, size_t len);
    void Append(DomainTag tag, const std::vector<uint8_t>& data);

    /**
     * @brief Append a 64-bit integer (for counters, indices)
     */
    void AppendU64(DomainTag tag, uint64_t value);

    /**
     * @brief Append a hash value
     */
    void AppendHash(DomainTag tag, const Hash256& hash);

    /**
     * @brief Append an LWE ciphertext
     */
    void AppendCiphertext(DomainTag tag, const LWECiphertext& ct);

    /**
     * @brief Append multiple ciphertexts (serialized sequentially)
     */
    void AppendCiphertexts(DomainTag tag, const std::vector<LWECiphertext>& cts);

    /**
     * @brief Append a NativeInteger
     */
    void AppendNativeInt(DomainTag tag, const NativeInteger& n);

    /**
     * @brief Append a NativeVector
     */
    void AppendNativeVector(DomainTag tag, const NativeVector& v);

    // ========================================================================
    // Challenge Generation
    // ========================================================================

    /**
     * @brief Generate a 256-bit challenge
     *
     * This finalizes the current transcript state and produces a challenge.
     * The transcript can continue to be used after this call.
     */
    Hash256 Challenge();

    /**
     * @brief Generate a challenge as NativeInteger mod q
     */
    NativeInteger ChallengeModQ(const NativeInteger& q);

    /**
     * @brief Generate multiple challenges (for batch verification)
     */
    std::vector<Hash256> Challenges(size_t count);

    /**
     * @brief Generate challenges as NativeIntegers
     */
    std::vector<NativeInteger> ChallengesModQ(size_t count, const NativeInteger& q);

    // ========================================================================
    // Transcript State
    // ========================================================================

    /**
     * @brief Clone current transcript state
     *
     * Useful for forking transcripts in parallel proof generation.
     */
    TranscriptBuilder Clone() const;

    /**
     * @brief Reset transcript to initial state (keeps protocol label)
     */
    void Reset();

    /**
     * @brief Get current transcript hash (without finalizing)
     */
    Hash256 CurrentHash() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// Merkle Tree for Batch Commitments
// ============================================================================

/**
 * @brief Merkle tree over batch elements
 *
 * INNOVATION: Instead of hashing each element into the transcript separately,
 * build a Merkle tree (parallelizable) and commit to the root.
 *
 * Properties:
 * - O(n) hashes total, but parallelizable across n leaves
 * - Single root hash for batch commitment
 * - Inclusion proofs for individual elements
 */
class MerkleTree {
public:
    MerkleTree();
    ~MerkleTree();

    // Non-copyable, movable
    MerkleTree(const MerkleTree&) = delete;
    MerkleTree& operator=(const MerkleTree&) = delete;
    MerkleTree(MerkleTree&&) noexcept;
    MerkleTree& operator=(MerkleTree&&) noexcept;

    // ========================================================================
    // Tree Construction
    // ========================================================================

    /**
     * @brief Build tree from leaf hashes
     *
     * Leaves are hashed with DomainTag::MERKLE_LEAF prefix.
     * Internal nodes use DomainTag::MERKLE_INTERNAL.
     *
     * @param leaves Pre-hashed leaf values
     */
    void Build(const std::vector<Hash256>& leaves);

    /**
     * @brief Build tree from raw data (computes leaf hashes)
     *
     * @param data Raw byte arrays for each leaf
     */
    void BuildFromData(const std::vector<std::vector<uint8_t>>& data);

    /**
     * @brief Build tree from LWE ciphertexts
     *
     * @param cts Ciphertexts to commit
     */
    void BuildFromCiphertexts(const std::vector<LWECiphertext>& cts);

    // ========================================================================
    // Tree Queries
    // ========================================================================

    /**
     * @brief Get the Merkle root
     */
    Hash256 Root() const;

    /**
     * @brief Get number of leaves
     */
    size_t NumLeaves() const;

    /**
     * @brief Get leaf hash at index
     */
    Hash256 LeafHash(size_t index) const;

    // ========================================================================
    // Inclusion Proofs
    // ========================================================================

    /**
     * @brief Merkle inclusion proof
     */
    struct InclusionProof {
        size_t leaf_index;
        Hash256 leaf_hash;
        std::vector<Hash256> siblings;  // Path from leaf to root
        std::vector<bool> path_bits;    // 0 = left child, 1 = right child
    };

    /**
     * @brief Generate inclusion proof for leaf
     */
    InclusionProof ProveInclusion(size_t leaf_index) const;

    /**
     * @brief Verify inclusion proof
     */
    static bool VerifyInclusion(
        const Hash256& root,
        const InclusionProof& proof
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// Batch Transcript
// ============================================================================

/**
 * @brief Optimized transcript for batch threshold operations
 *
 * INNOVATION: Combines Merkle tree commitment with challenge derivation:
 *
 * Traditional (per-ciphertext):
 *   for i in 0..n:
 *       transcript.append(ct[i])
 *       challenges[i] = transcript.hash()  // n hashes, SERIAL
 *
 * Batched (our approach):
 *   merkle.build(cts)                       // n hashes, PARALLEL
 *   batch_challenge = hash(merkle.root())   // 1 hash
 *   for i in 0..n:
 *       challenges[i] = derive(batch_challenge, i)  // n PRF calls, PARALLEL
 *
 * The PRF derivation uses SHAKE256 for efficiency.
 */
class BatchTranscript {
public:
    /**
     * @brief Create batch transcript
     * @param protocol_label Protocol identifier
     */
    explicit BatchTranscript(const std::string& protocol_label);
    ~BatchTranscript();

    // Non-copyable, movable
    BatchTranscript(const BatchTranscript&) = delete;
    BatchTranscript& operator=(const BatchTranscript&) = delete;
    BatchTranscript(BatchTranscript&&) noexcept;
    BatchTranscript& operator=(BatchTranscript&&) noexcept;

    // ========================================================================
    // Batch Commitment
    // ========================================================================

    /**
     * @brief Commit to a batch of ciphertexts
     *
     * Builds Merkle tree and appends root to transcript.
     * Returns the Merkle root for verification.
     */
    Hash256 CommitBatch(const std::vector<LWECiphertext>& cts);

    /**
     * @brief Commit to partial decryptions
     *
     * @param party_id ID of the decrypting party
     * @param partials Partial decryption values
     */
    Hash256 CommitPartialDecryptions(
        uint32_t party_id,
        const std::vector<NativeInteger>& partials
    );

    /**
     * @brief Commit to key shares
     */
    Hash256 CommitKeyShares(
        uint32_t party_id,
        const NativeVector& share
    );

    // ========================================================================
    // Challenge Generation
    // ========================================================================

    /**
     * @brief Generate batch challenge (single hash of transcript state)
     */
    Hash256 BatchChallenge();

    /**
     * @brief Derive per-element challenges from batch challenge
     *
     * @param count Number of challenges to derive
     * @return Vector of challenges, one per batch element
     */
    std::vector<Hash256> DeriveElementChallenges(size_t count);

    /**
     * @brief Derive challenges as field elements mod q
     */
    std::vector<NativeInteger> DeriveElementChallengesModQ(
        size_t count,
        const NativeInteger& q
    );

    // ========================================================================
    // Auxiliary Data
    // ========================================================================

    /**
     * @brief Append auxiliary public data to transcript
     *
     * Use for protocol-specific public parameters.
     */
    void AppendPublicData(const std::string& label, const std::vector<uint8_t>& data);

    /**
     * @brief Append party information
     */
    void AppendPartyInfo(uint32_t party_id, uint32_t threshold, uint32_t total_parties);

    // ========================================================================
    // State Access
    // ========================================================================

    /**
     * @brief Get the underlying Merkle tree (for inclusion proofs)
     */
    const MerkleTree& GetMerkleTree() const;

    /**
     * @brief Reset transcript for new batch
     */
    void Reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// Hash Functions (exposed for testing)
// ============================================================================

namespace hash {

/**
 * @brief SHA3-256 hash
 */
Hash256 SHA3_256(const uint8_t* data, size_t len);
Hash256 SHA3_256(const std::vector<uint8_t>& data);

/**
 * @brief SHAKE256 variable-length output
 */
void SHAKE256(const uint8_t* data, size_t data_len, uint8_t* out, size_t out_len);
std::vector<uint8_t> SHAKE256(const std::vector<uint8_t>& data, size_t out_len);

/**
 * @brief Hash with domain separation
 */
Hash256 HashWithDomain(DomainTag tag, const uint8_t* data, size_t len);

/**
 * @brief Merkle hash of two children
 */
Hash256 MerkleHash(const Hash256& left, const Hash256& right);

/**
 * @brief Derive challenge from seed and index (for batch challenge derivation)
 */
Hash256 DeriveChallenge(const Hash256& seed, uint64_t index);

/**
 * @brief Convert hash to NativeInteger mod q
 */
NativeInteger HashToFieldElement(const Hash256& h, const NativeInteger& q);

} // namespace hash

} // namespace threshold
} // namespace lux::fhe

#endif // THRESHOLD_TRANSCRIPT_H
