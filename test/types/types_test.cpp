// =============================================================================
// Lux FHE - Type System Layout & Validation Tests
// =============================================================================
//
// Verifies that FHECiphertextHeader and FHEPrecompileArtifact match the Go
// definitions in github.com/luxfi/fhe/types byte-for-byte.
//
// Reference: LP-137-FHE-TYPING.md.
//
// Build (standalone, no gtest required):
//
//   c++ -std=c++17 -I include -I ../lattice/include test/types/types_test.cpp \
//       -o types_test && ./types_test
//
// Copyright (C) 2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
// =============================================================================

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "lux/fhe/types/artifact.hpp"
#include "lux/fhe/types/ciphertext_header.hpp"
#include "lux/lattice/types/domain.hpp"

using lux::fhe::FHECiphertextHeader;
using lux::fhe::FHEPrecompileArtifact;
using lux::fhe::FHEScheme;
using lux::lattice::PolyDomain;

namespace {

int tests_run = 0;
int tests_failed = 0;

#define CHECK(cond)                                              \
    do {                                                         \
        ++tests_run;                                             \
        if (!(cond)) {                                           \
            ++tests_failed;                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n",             \
                         __FILE__, __LINE__, #cond);             \
        }                                                        \
    } while (0)

// -----------------------------------------------------------------------------
// FHEScheme enum values stable on the wire
// -----------------------------------------------------------------------------
void test_fhe_scheme_values() {
    CHECK(static_cast<std::uint32_t>(FHEScheme::TFHE) == 0);
    CHECK(static_cast<std::uint32_t>(FHEScheme::FHEW) == 1);
    CHECK(static_cast<std::uint32_t>(FHEScheme::CKKS) == 2);
    CHECK(static_cast<std::uint32_t>(FHEScheme::BFV)  == 3);
    CHECK(static_cast<std::uint32_t>(FHEScheme::BGV)  == 4);
}

// -----------------------------------------------------------------------------
// FHECiphertextHeader layout
// -----------------------------------------------------------------------------
void test_ciphertext_header_layout() {
    CHECK(sizeof(FHECiphertextHeader) == 144);
    CHECK(offsetof(FHECiphertextHeader, params_hash)   == 0);
    CHECK(offsetof(FHECiphertextHeader, key_id)        == 32);
    CHECK(offsetof(FHECiphertextHeader, circuit_id)    == 64);
    CHECK(offsetof(FHECiphertextHeader, scheme)        == 96);
    CHECK(offsetof(FHECiphertextHeader, level)         == 100);
    CHECK(offsetof(FHECiphertextHeader, N)             == 104);
    CHECK(offsetof(FHECiphertextHeader, modulus_count) == 108);
    CHECK(offsetof(FHECiphertextHeader, domain)        == 112);
    CHECK(offsetof(FHECiphertextHeader, reserved)      == 120);
}

// memcmp test: build a 144-byte canonical image, lay an FHECiphertextHeader
// on top, and verify field values reconstruct correctly. Round-trip via
// memcpy must match. This is the cross-language invariant.
void test_ciphertext_header_memcmp() {
    std::uint8_t want[144];
    for (int i = 0; i < 32; i++)  want[i]       = static_cast<std::uint8_t>(i);
    for (int i = 0; i < 32; i++)  want[32 + i]  = static_cast<std::uint8_t>(i + 0x40);
    for (int i = 0; i < 32; i++)  want[64 + i]  = static_cast<std::uint8_t>(i + 0x80);
    // scheme = CKKS (2)
    want[96] = 0x02; want[97] = 0x00; want[98] = 0x00; want[99] = 0x00;
    // level = 7
    want[100] = 0x07; want[101] = 0x00; want[102] = 0x00; want[103] = 0x00;
    // N = 16384 = 0x4000
    want[104] = 0x00; want[105] = 0x40; want[106] = 0x00; want[107] = 0x00;
    // modulus_count = 8
    want[108] = 0x08; want[109] = 0x00; want[110] = 0x00; want[111] = 0x00;
    // domain = NTTMontgomery (3)
    want[112] = 0x03;
    // pad bytes 113..119 = 0
    for (int i = 113; i < 120; i++) want[i] = 0;
    // reserved 120..143 = 0
    for (int i = 120; i < 144; i++) want[i] = 0;

    FHECiphertextHeader h{};
    std::memcpy(&h, want, sizeof(want));

    for (int i = 0; i < 32; i++) {
        CHECK(h.params_hash[i] == static_cast<std::uint8_t>(i));
        CHECK(h.key_id[i]      == static_cast<std::uint8_t>(i + 0x40));
        CHECK(h.circuit_id[i]  == static_cast<std::uint8_t>(i + 0x80));
    }
    CHECK(h.scheme        == FHEScheme::CKKS);
    CHECK(h.level         == 7u);
    CHECK(h.N             == 16384u);
    CHECK(h.modulus_count == 8u);
    CHECK(h.domain        == PolyDomain::NTTMontgomery);

    // Round-trip
    std::uint8_t got[144];
    std::memcpy(got, &h, sizeof(h));
    CHECK(std::memcmp(got, want, sizeof(want)) == 0);
}

// -----------------------------------------------------------------------------
// FHEPrecompileArtifact layout
// -----------------------------------------------------------------------------
void test_artifact_layout() {
    CHECK(sizeof(FHEPrecompileArtifact) == 232);
    CHECK(offsetof(FHEPrecompileArtifact, params_hash)               == 0);
    CHECK(offsetof(FHEPrecompileArtifact, key_root)                  == 32);
    CHECK(offsetof(FHEPrecompileArtifact, input_ciphertext_root)     == 64);
    CHECK(offsetof(FHEPrecompileArtifact, output_ciphertext_root)    == 96);
    CHECK(offsetof(FHEPrecompileArtifact, circuit_root)              == 128);
    CHECK(offsetof(FHEPrecompileArtifact, threshold_transcript_root) == 160);
    CHECK(offsetof(FHEPrecompileArtifact, attestation_root)          == 192);
    CHECK(offsetof(FHEPrecompileArtifact, op_count)                  == 224);
    CHECK(offsetof(FHEPrecompileArtifact, failed_count)              == 228);
}

void test_artifact_memcmp() {
    std::uint8_t want[232]{};
    for (int i = 0; i < 32; i++) want[i]       = static_cast<std::uint8_t>(i);
    for (int i = 0; i < 32; i++) want[32 + i]  = static_cast<std::uint8_t>(i + 1);
    for (int i = 0; i < 32; i++) want[64 + i]  = static_cast<std::uint8_t>(i + 2);
    for (int i = 0; i < 32; i++) want[96 + i]  = static_cast<std::uint8_t>(i + 3);
    for (int i = 0; i < 32; i++) want[128 + i] = static_cast<std::uint8_t>(i + 4);
    for (int i = 0; i < 32; i++) want[160 + i] = static_cast<std::uint8_t>(i + 5);
    for (int i = 0; i < 32; i++) want[192 + i] = static_cast<std::uint8_t>(i + 6);
    // op_count = 1024
    want[224] = 0x00; want[225] = 0x04; want[226] = 0x00; want[227] = 0x00;
    // failed_count = 0
    want[228] = 0; want[229] = 0; want[230] = 0; want[231] = 0;

    FHEPrecompileArtifact a{};
    std::memcpy(&a, want, sizeof(want));

    for (int i = 0; i < 32; i++) {
        CHECK(a.params_hash[i]               == static_cast<std::uint8_t>(i));
        CHECK(a.key_root[i]                  == static_cast<std::uint8_t>(i + 1));
        CHECK(a.input_ciphertext_root[i]     == static_cast<std::uint8_t>(i + 2));
        CHECK(a.output_ciphertext_root[i]    == static_cast<std::uint8_t>(i + 3));
        CHECK(a.circuit_root[i]              == static_cast<std::uint8_t>(i + 4));
        CHECK(a.threshold_transcript_root[i] == static_cast<std::uint8_t>(i + 5));
        CHECK(a.attestation_root[i]          == static_cast<std::uint8_t>(i + 6));
    }
    CHECK(a.op_count     == 1024u);
    CHECK(a.failed_count == 0u);

    std::uint8_t got[232];
    std::memcpy(got, &a, sizeof(a));
    CHECK(std::memcmp(got, want, sizeof(want)) == 0);
}

} // namespace

int main() {
    test_fhe_scheme_values();
    test_ciphertext_header_layout();
    test_ciphertext_header_memcmp();
    test_artifact_layout();
    test_artifact_memcmp();

    std::printf("ran %d checks, %d failed\n", tests_run, tests_failed);
    return tests_failed == 0 ? 0 : 1;
}
