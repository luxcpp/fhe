// =============================================================================
// Lux FHE - FHEPrecompileArtifact (C++ mirror of luxfi/fhe/types)
// =============================================================================
//
// Layout-stable mirror of github.com/luxfi/fhe/types artifact.go.
//
// Reference: LP-137-FHE-TYPING.md.
//
// Copyright (C) 2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
// =============================================================================

#ifndef LUX_FHE_TYPES_ARTIFACT_HPP
#define LUX_FHE_TYPES_ARTIFACT_HPP

#include <cstddef>
#include <cstdint>

namespace lux::fhe {

// FHEPrecompileArtifact is the fixed-size record produced by an FHE
// precompile invocation, consumed by the QuasarGPU integration layer
// (LP-132) and the FChainTFHE cert lane (LP-013, LP-020 §3.0).
//
// Total size: 232 bytes.
struct FHEPrecompileArtifact {
    std::uint8_t  params_hash[32];                 // offset 0,   size 32
    std::uint8_t  key_root[32];                    // offset 32,  size 32
    std::uint8_t  input_ciphertext_root[32];       // offset 64,  size 32
    std::uint8_t  output_ciphertext_root[32];      // offset 96,  size 32
    std::uint8_t  circuit_root[32];                // offset 128, size 32
    std::uint8_t  threshold_transcript_root[32];   // offset 160, size 32
    std::uint8_t  attestation_root[32];            // offset 192, size 32
    std::uint32_t op_count;                        // offset 224, size 4
    std::uint32_t failed_count;                    // offset 228, size 4
};

static_assert(sizeof(FHEPrecompileArtifact) == 232,
              "FHEPrecompileArtifact must be 232 bytes");
static_assert(offsetof(FHEPrecompileArtifact, params_hash)               == 0,
              "params_hash offset");
static_assert(offsetof(FHEPrecompileArtifact, key_root)                  == 32,
              "key_root offset");
static_assert(offsetof(FHEPrecompileArtifact, input_ciphertext_root)     == 64,
              "input_ciphertext_root offset");
static_assert(offsetof(FHEPrecompileArtifact, output_ciphertext_root)    == 96,
              "output_ciphertext_root offset");
static_assert(offsetof(FHEPrecompileArtifact, circuit_root)              == 128,
              "circuit_root offset");
static_assert(offsetof(FHEPrecompileArtifact, threshold_transcript_root) == 160,
              "threshold_transcript_root offset");
static_assert(offsetof(FHEPrecompileArtifact, attestation_root)          == 192,
              "attestation_root offset");
static_assert(offsetof(FHEPrecompileArtifact, op_count)                  == 224,
              "op_count offset");
static_assert(offsetof(FHEPrecompileArtifact, failed_count)              == 228,
              "failed_count offset");

} // namespace lux::fhe

#endif // LUX_FHE_TYPES_ARTIFACT_HPP
