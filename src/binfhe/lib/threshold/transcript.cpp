// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// Fiat-Shamir Transcript Implementation

#include "threshold/transcript.h"
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <iomanip>

// For SHA3/SHAKE256, we use a portable implementation
// In production, this would use OpenSSL or a hardware-accelerated library
#include <array>

namespace lux::fhe {
namespace threshold {

// ============================================================================
// Hash256 Operations
// ============================================================================

bool operator==(const Hash256& a, const Hash256& b) {
    // Constant-time comparison to prevent timing attacks
    uint8_t diff = 0;
    for (size_t i = 0; i < 32; ++i) {
        diff |= a[i] ^ b[i];
    }
    return diff == 0;
}

bool operator!=(const Hash256& a, const Hash256& b) {
    return !(a == b);
}

std::string HashToHex(const Hash256& h) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (size_t i = 0; i < 32; ++i) {
        oss << std::setw(2) << static_cast<int>(h[i]);
    }
    return oss.str();
}

Hash256 HexToHash(const std::string& hex) {
    Hash256 h{};
    if (hex.length() != 64) {
        throw std::invalid_argument("Hex string must be 64 characters");
    }
    for (size_t i = 0; i < 32; ++i) {
        h[i] = static_cast<uint8_t>(std::stoul(hex.substr(i * 2, 2), nullptr, 16));
    }
    return h;
}

// ============================================================================
// Keccak Implementation (for SHA3/SHAKE256)
// ============================================================================

namespace {

// Keccak-f[1600] constants
constexpr uint64_t keccak_rc[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

constexpr int keccak_rotc[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};

constexpr int keccak_piln[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};

inline uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

void keccak_f1600(uint64_t st[25]) {
    uint64_t t, bc[5];

    for (int round = 0; round < 24; round++) {
        // Theta
        for (int i = 0; i < 5; i++) {
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        }
        for (int i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5) {
                st[j + i] ^= t;
            }
        }

        // Rho Pi
        t = st[1];
        for (int i = 0; i < 24; i++) {
            int j = keccak_piln[i];
            bc[0] = st[j];
            st[j] = rotl64(t, keccak_rotc[i]);
            t = bc[0];
        }

        // Chi
        for (int j = 0; j < 25; j += 5) {
            for (int i = 0; i < 5; i++) {
                bc[i] = st[j + i];
            }
            for (int i = 0; i < 5; i++) {
                st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
            }
        }

        // Iota
        st[0] ^= keccak_rc[round];
    }
}

class KeccakState {
public:
    KeccakState(size_t rate_bytes) : rate_(rate_bytes), offset_(0) {
        std::memset(state_, 0, sizeof(state_));
    }

    void Absorb(const uint8_t* data, size_t len) {
        uint8_t* st_bytes = reinterpret_cast<uint8_t*>(state_);

        while (len > 0) {
            size_t to_absorb = std::min(len, rate_ - offset_);
            for (size_t i = 0; i < to_absorb; i++) {
                st_bytes[offset_ + i] ^= data[i];
            }
            offset_ += to_absorb;
            data += to_absorb;
            len -= to_absorb;

            if (offset_ == rate_) {
                keccak_f1600(state_);
                offset_ = 0;
            }
        }
    }

    void Finalize(uint8_t pad_byte) {
        uint8_t* st_bytes = reinterpret_cast<uint8_t*>(state_);
        st_bytes[offset_] ^= pad_byte;
        st_bytes[rate_ - 1] ^= 0x80;
        keccak_f1600(state_);
        offset_ = 0;
    }

    void Squeeze(uint8_t* out, size_t out_len) {
        uint8_t* st_bytes = reinterpret_cast<uint8_t*>(state_);

        while (out_len > 0) {
            if (offset_ == rate_) {
                keccak_f1600(state_);
                offset_ = 0;
            }
            size_t to_squeeze = std::min(out_len, rate_ - offset_);
            std::memcpy(out, st_bytes + offset_, to_squeeze);
            offset_ += to_squeeze;
            out += to_squeeze;
            out_len -= to_squeeze;
        }
    }

private:
    uint64_t state_[25];
    size_t rate_;
    size_t offset_;
};

} // anonymous namespace

// ============================================================================
// Hash Functions
// ============================================================================

namespace hash {

Hash256 SHA3_256(const uint8_t* data, size_t len) {
    // SHA3-256: rate = 1088 bits = 136 bytes, capacity = 512 bits
    KeccakState state(136);
    state.Absorb(data, len);
    state.Finalize(0x06);  // SHA3 domain separator

    Hash256 result;
    state.Squeeze(result.data(), 32);
    return result;
}

Hash256 SHA3_256(const std::vector<uint8_t>& data) {
    return SHA3_256(data.data(), data.size());
}

void SHAKE256(const uint8_t* data, size_t data_len, uint8_t* out, size_t out_len) {
    // SHAKE256: rate = 1088 bits = 136 bytes
    KeccakState state(136);
    state.Absorb(data, data_len);
    state.Finalize(0x1F);  // SHAKE domain separator
    state.Squeeze(out, out_len);
}

std::vector<uint8_t> SHAKE256(const std::vector<uint8_t>& data, size_t out_len) {
    std::vector<uint8_t> out(out_len);
    SHAKE256(data.data(), data.size(), out.data(), out_len);
    return out;
}

Hash256 HashWithDomain(DomainTag tag, const uint8_t* data, size_t len) {
    std::vector<uint8_t> prefixed(1 + len);
    prefixed[0] = static_cast<uint8_t>(tag);
    std::memcpy(prefixed.data() + 1, data, len);
    return SHA3_256(prefixed);
}

Hash256 MerkleHash(const Hash256& left, const Hash256& right) {
    std::vector<uint8_t> combined(1 + 64);
    combined[0] = static_cast<uint8_t>(DomainTag::MERKLE_INTERNAL);
    std::memcpy(combined.data() + 1, left.data(), 32);
    std::memcpy(combined.data() + 33, right.data(), 32);
    return SHA3_256(combined);
}

Hash256 DeriveChallenge(const Hash256& seed, uint64_t index) {
    std::vector<uint8_t> input(1 + 32 + 8);
    input[0] = static_cast<uint8_t>(DomainTag::ELEMENT_CHALLENGE);
    std::memcpy(input.data() + 1, seed.data(), 32);
    // Little-endian index
    for (int i = 0; i < 8; i++) {
        input[33 + i] = static_cast<uint8_t>(index >> (8 * i));
    }
    return SHA3_256(input);
}

NativeInteger HashToFieldElement(const Hash256& h, const NativeInteger& q) {
    // Interpret hash as big-endian integer and reduce mod q
    // Use rejection sampling for uniformity

    // For simplicity, we use the first 128 bits and reduce
    // This is slightly biased but acceptable for most applications
    NativeInteger result(0);
    NativeInteger base(1);
    NativeInteger byte_mult(256);

    // Process 16 bytes (128 bits) for NativeInteger range
    for (int i = 0; i < 16 && i < 32; i++) {
        result = result.ModAdd(base.ModMul(NativeInteger(h[31 - i]), q), q);
        base = base.ModMul(byte_mult, q);
    }

    return result;
}

} // namespace hash

// ============================================================================
// TranscriptBuilder Implementation
// ============================================================================

struct TranscriptBuilder::Impl {
    std::string protocol_label;
    std::vector<uint8_t> buffer;

    Impl(const std::string& label) : protocol_label(label) {
        // Initialize with protocol label
        AppendBytes(reinterpret_cast<const uint8_t*>(label.data()), label.size());
        AppendU64(static_cast<uint64_t>(label.size()));
    }

    void AppendBytes(const uint8_t* data, size_t len) {
        buffer.insert(buffer.end(), data, data + len);
    }

    void AppendU64(uint64_t val) {
        uint8_t bytes[8];
        for (int i = 0; i < 8; i++) {
            bytes[i] = static_cast<uint8_t>(val >> (8 * i));
        }
        AppendBytes(bytes, 8);
    }
};

TranscriptBuilder::TranscriptBuilder(const std::string& protocol_label)
    : impl_(std::make_unique<Impl>(protocol_label)) {}

TranscriptBuilder::~TranscriptBuilder() = default;

TranscriptBuilder::TranscriptBuilder(TranscriptBuilder&&) noexcept = default;
TranscriptBuilder& TranscriptBuilder::operator=(TranscriptBuilder&&) noexcept = default;

void TranscriptBuilder::Append(DomainTag tag, const uint8_t* data, size_t len) {
    impl_->buffer.push_back(static_cast<uint8_t>(tag));
    impl_->AppendU64(len);
    impl_->AppendBytes(data, len);
}

void TranscriptBuilder::Append(DomainTag tag, const std::vector<uint8_t>& data) {
    Append(tag, data.data(), data.size());
}

void TranscriptBuilder::AppendU64(DomainTag tag, uint64_t value) {
    impl_->buffer.push_back(static_cast<uint8_t>(tag));
    impl_->AppendU64(value);
}

void TranscriptBuilder::AppendHash(DomainTag tag, const Hash256& h) {
    Append(tag, h.data(), 32);
}

void TranscriptBuilder::AppendCiphertext(DomainTag tag, const LWECiphertext& ct) {
    // Serialize ciphertext
    const NativeVector& a = ct->GetA();
    const NativeInteger& b = ct->GetB();
    const NativeInteger& q = ct->GetModulus();

    std::vector<uint8_t> serialized;

    // Append modulus
    uint64_t q_val = q.ConvertToInt<uint64_t>();
    for (int i = 0; i < 8; i++) {
        serialized.push_back(static_cast<uint8_t>(q_val >> (8 * i)));
    }

    // Append dimension
    uint32_t n = a.GetLength();
    for (int i = 0; i < 4; i++) {
        serialized.push_back(static_cast<uint8_t>(n >> (8 * i)));
    }

    // Append a vector
    for (uint32_t i = 0; i < n; i++) {
        uint64_t val = a[i].ConvertToInt<uint64_t>();
        for (int j = 0; j < 8; j++) {
            serialized.push_back(static_cast<uint8_t>(val >> (8 * j)));
        }
    }

    // Append b
    uint64_t b_val = b.ConvertToInt<uint64_t>();
    for (int i = 0; i < 8; i++) {
        serialized.push_back(static_cast<uint8_t>(b_val >> (8 * i)));
    }

    Append(tag, serialized);
}

void TranscriptBuilder::AppendCiphertexts(DomainTag tag, const std::vector<LWECiphertext>& cts) {
    AppendU64(tag, cts.size());
    for (const auto& ct : cts) {
        AppendCiphertext(tag, ct);
    }
}

void TranscriptBuilder::AppendNativeInt(DomainTag tag, const NativeInteger& n) {
    uint64_t val = n.ConvertToInt<uint64_t>();
    uint8_t bytes[8];
    for (int i = 0; i < 8; i++) {
        bytes[i] = static_cast<uint8_t>(val >> (8 * i));
    }
    Append(tag, bytes, 8);
}

void TranscriptBuilder::AppendNativeVector(DomainTag tag, const NativeVector& v) {
    std::vector<uint8_t> serialized;
    uint32_t len = v.GetLength();
    for (int i = 0; i < 4; i++) {
        serialized.push_back(static_cast<uint8_t>(len >> (8 * i)));
    }
    for (uint32_t i = 0; i < len; i++) {
        uint64_t val = v[i].ConvertToInt<uint64_t>();
        for (int j = 0; j < 8; j++) {
            serialized.push_back(static_cast<uint8_t>(val >> (8 * j)));
        }
    }
    Append(tag, serialized);
}

Hash256 TranscriptBuilder::Challenge() {
    return hash::SHA3_256(impl_->buffer);
}

NativeInteger TranscriptBuilder::ChallengeModQ(const NativeInteger& q) {
    Hash256 h = Challenge();
    return hash::HashToFieldElement(h, q);
}

std::vector<Hash256> TranscriptBuilder::Challenges(size_t count) {
    std::vector<Hash256> result(count);
    Hash256 seed = Challenge();
    for (size_t i = 0; i < count; i++) {
        result[i] = hash::DeriveChallenge(seed, i);
    }
    return result;
}

std::vector<NativeInteger> TranscriptBuilder::ChallengesModQ(size_t count, const NativeInteger& q) {
    std::vector<NativeInteger> result(count);
    Hash256 seed = Challenge();
    for (size_t i = 0; i < count; i++) {
        Hash256 h = hash::DeriveChallenge(seed, i);
        result[i] = hash::HashToFieldElement(h, q);
    }
    return result;
}

TranscriptBuilder TranscriptBuilder::Clone() const {
    TranscriptBuilder clone(impl_->protocol_label);
    clone.impl_->buffer = impl_->buffer;
    return clone;
}

void TranscriptBuilder::Reset() {
    impl_->buffer.clear();
    impl_->AppendBytes(
        reinterpret_cast<const uint8_t*>(impl_->protocol_label.data()),
        impl_->protocol_label.size()
    );
    impl_->AppendU64(static_cast<uint64_t>(impl_->protocol_label.size()));
}

Hash256 TranscriptBuilder::CurrentHash() const {
    return hash::SHA3_256(impl_->buffer);
}

// ============================================================================
// MerkleTree Implementation
// ============================================================================

struct MerkleTree::Impl {
    std::vector<Hash256> leaves;
    std::vector<std::vector<Hash256>> levels;  // levels[0] = leaves, levels[last] = root

    void BuildLevels() {
        if (leaves.empty()) {
            levels.clear();
            return;
        }

        levels.clear();
        levels.push_back(leaves);

        while (levels.back().size() > 1) {
            const auto& prev = levels.back();
            std::vector<Hash256> next;
            next.reserve((prev.size() + 1) / 2);

            for (size_t i = 0; i < prev.size(); i += 2) {
                if (i + 1 < prev.size()) {
                    next.push_back(hash::MerkleHash(prev[i], prev[i + 1]));
                } else {
                    // Odd number: hash with itself
                    next.push_back(hash::MerkleHash(prev[i], prev[i]));
                }
            }
            levels.push_back(std::move(next));
        }
    }
};

MerkleTree::MerkleTree() : impl_(std::make_unique<Impl>()) {}
MerkleTree::~MerkleTree() = default;

MerkleTree::MerkleTree(MerkleTree&&) noexcept = default;
MerkleTree& MerkleTree::operator=(MerkleTree&&) noexcept = default;

void MerkleTree::Build(const std::vector<Hash256>& leaves) {
    impl_->leaves = leaves;
    impl_->BuildLevels();
}

void MerkleTree::BuildFromData(const std::vector<std::vector<uint8_t>>& data) {
    impl_->leaves.resize(data.size());

    // Parallelize leaf hashing
    #pragma omp parallel for if(data.size() > 16)
    for (size_t i = 0; i < data.size(); i++) {
        impl_->leaves[i] = hash::HashWithDomain(DomainTag::MERKLE_LEAF, data[i].data(), data[i].size());
    }

    impl_->BuildLevels();
}

void MerkleTree::BuildFromCiphertexts(const std::vector<LWECiphertext>& cts) {
    // Convert ciphertexts to byte arrays
    std::vector<std::vector<uint8_t>> data(cts.size());

    #pragma omp parallel for if(cts.size() > 16)
    for (size_t i = 0; i < cts.size(); i++) {
        const NativeVector& a = cts[i]->GetA();
        const NativeInteger& b = cts[i]->GetB();
        uint32_t n = a.GetLength();

        data[i].resize(4 + 8 * n + 8);

        // Dimension
        for (int j = 0; j < 4; j++) {
            data[i][j] = static_cast<uint8_t>(n >> (8 * j));
        }

        // a vector
        for (uint32_t j = 0; j < n; j++) {
            uint64_t val = a[j].ConvertToInt<uint64_t>();
            for (int k = 0; k < 8; k++) {
                data[i][4 + 8 * j + k] = static_cast<uint8_t>(val >> (8 * k));
            }
        }

        // b
        uint64_t b_val = b.ConvertToInt<uint64_t>();
        for (int j = 0; j < 8; j++) {
            data[i][4 + 8 * n + j] = static_cast<uint8_t>(b_val >> (8 * j));
        }
    }

    BuildFromData(data);
}

Hash256 MerkleTree::Root() const {
    if (impl_->levels.empty()) {
        return Hash256{};
    }
    return impl_->levels.back()[0];
}

size_t MerkleTree::NumLeaves() const {
    return impl_->leaves.size();
}

Hash256 MerkleTree::LeafHash(size_t index) const {
    if (index >= impl_->leaves.size()) {
        throw std::out_of_range("Leaf index out of range");
    }
    return impl_->leaves[index];
}

MerkleTree::InclusionProof MerkleTree::ProveInclusion(size_t leaf_index) const {
    if (leaf_index >= impl_->leaves.size()) {
        throw std::out_of_range("Leaf index out of range");
    }

    InclusionProof proof;
    proof.leaf_index = leaf_index;
    proof.leaf_hash = impl_->leaves[leaf_index];

    size_t idx = leaf_index;
    for (size_t level = 0; level + 1 < impl_->levels.size(); level++) {
        const auto& current_level = impl_->levels[level];
        size_t sibling_idx = (idx % 2 == 0) ? idx + 1 : idx - 1;

        if (sibling_idx < current_level.size()) {
            proof.siblings.push_back(current_level[sibling_idx]);
        } else {
            // Odd number of nodes: sibling is self
            proof.siblings.push_back(current_level[idx]);
        }

        proof.path_bits.push_back(idx % 2 == 1);  // true if we're right child
        idx /= 2;
    }

    return proof;
}

bool MerkleTree::VerifyInclusion(const Hash256& root, const InclusionProof& proof) {
    Hash256 current = proof.leaf_hash;

    for (size_t i = 0; i < proof.siblings.size(); i++) {
        if (proof.path_bits[i]) {
            // We're right child
            current = hash::MerkleHash(proof.siblings[i], current);
        } else {
            // We're left child
            current = hash::MerkleHash(current, proof.siblings[i]);
        }
    }

    return current == root;
}

// ============================================================================
// BatchTranscript Implementation
// ============================================================================

struct BatchTranscript::Impl {
    std::string protocol_label;
    TranscriptBuilder transcript;
    MerkleTree merkle_tree;
    Hash256 batch_challenge;
    bool challenge_computed;

    Impl(const std::string& label)
        : protocol_label(label)
        , transcript(label)
        , challenge_computed(false) {}
};

BatchTranscript::BatchTranscript(const std::string& protocol_label)
    : impl_(std::make_unique<Impl>(protocol_label)) {}

BatchTranscript::~BatchTranscript() = default;

BatchTranscript::BatchTranscript(BatchTranscript&&) noexcept = default;
BatchTranscript& BatchTranscript::operator=(BatchTranscript&&) noexcept = default;

Hash256 BatchTranscript::CommitBatch(const std::vector<LWECiphertext>& cts) {
    impl_->merkle_tree.BuildFromCiphertexts(cts);
    Hash256 root = impl_->merkle_tree.Root();
    impl_->transcript.AppendHash(DomainTag::BATCH_ROOT, root);
    impl_->challenge_computed = false;
    return root;
}

Hash256 BatchTranscript::CommitPartialDecryptions(
    uint32_t party_id,
    const std::vector<NativeInteger>& partials
) {
    // Build Merkle tree over partial decryptions
    std::vector<std::vector<uint8_t>> data(partials.size());

    #pragma omp parallel for if(partials.size() > 16)
    for (size_t i = 0; i < partials.size(); i++) {
        data[i].resize(12);  // 4 bytes party_id + 8 bytes value

        for (int j = 0; j < 4; j++) {
            data[i][j] = static_cast<uint8_t>(party_id >> (8 * j));
        }

        uint64_t val = partials[i].ConvertToInt<uint64_t>();
        for (int j = 0; j < 8; j++) {
            data[i][4 + j] = static_cast<uint8_t>(val >> (8 * j));
        }
    }

    MerkleTree tree;
    tree.BuildFromData(data);
    Hash256 root = tree.Root();

    impl_->transcript.AppendU64(DomainTag::PARTIAL_DECRYPT, party_id);
    impl_->transcript.AppendHash(DomainTag::PARTIAL_DECRYPT, root);
    impl_->challenge_computed = false;

    return root;
}

Hash256 BatchTranscript::CommitKeyShares(uint32_t party_id, const NativeVector& share) {
    impl_->transcript.AppendU64(DomainTag::KEY_SHARE, party_id);
    impl_->transcript.AppendNativeVector(DomainTag::KEY_SHARE, share);
    impl_->challenge_computed = false;
    return impl_->transcript.CurrentHash();
}

Hash256 BatchTranscript::BatchChallenge() {
    if (!impl_->challenge_computed) {
        impl_->batch_challenge = impl_->transcript.Challenge();
        impl_->challenge_computed = true;
    }
    return impl_->batch_challenge;
}

std::vector<Hash256> BatchTranscript::DeriveElementChallenges(size_t count) {
    Hash256 seed = BatchChallenge();
    std::vector<Hash256> result(count);

    #pragma omp parallel for if(count > 16)
    for (size_t i = 0; i < count; i++) {
        result[i] = hash::DeriveChallenge(seed, i);
    }

    return result;
}

std::vector<NativeInteger> BatchTranscript::DeriveElementChallengesModQ(
    size_t count,
    const NativeInteger& q
) {
    Hash256 seed = BatchChallenge();
    std::vector<NativeInteger> result(count);

    #pragma omp parallel for if(count > 16)
    for (size_t i = 0; i < count; i++) {
        Hash256 h = hash::DeriveChallenge(seed, i);
        result[i] = hash::HashToFieldElement(h, q);
    }

    return result;
}

void BatchTranscript::AppendPublicData(const std::string& label, const std::vector<uint8_t>& data) {
    // Append label as bytes
    impl_->transcript.Append(
        DomainTag::COMMITMENT,
        reinterpret_cast<const uint8_t*>(label.data()),
        label.size()
    );
    impl_->transcript.Append(DomainTag::COMMITMENT, data);
    impl_->challenge_computed = false;
}

void BatchTranscript::AppendPartyInfo(uint32_t party_id, uint32_t threshold, uint32_t total_parties) {
    impl_->transcript.AppendU64(DomainTag::KEY_SHARE, party_id);
    impl_->transcript.AppendU64(DomainTag::KEY_SHARE, threshold);
    impl_->transcript.AppendU64(DomainTag::KEY_SHARE, total_parties);
    impl_->challenge_computed = false;
}

const MerkleTree& BatchTranscript::GetMerkleTree() const {
    return impl_->merkle_tree;
}

void BatchTranscript::Reset() {
    impl_->transcript.Reset();
    impl_->merkle_tree = MerkleTree();
    impl_->challenge_computed = false;
}

} // namespace threshold
} // namespace lux::fhe
