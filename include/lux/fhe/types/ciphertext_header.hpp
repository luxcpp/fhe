// =============================================================================
// Lux FHE - FHEScheme + FHECiphertextHeader (C++ mirror of luxfi/fhe/types)
// =============================================================================
//
// Layout-stable mirror of github.com/luxfi/fhe/types ciphertext_header.go.
// Shared by cgo and C++ kernels for FHE-GPU dispatch.
//
// Reference: LP-137-FHE-TYPING.md.
//
// IMPORTANT: This struct must remain byte-for-byte identical to the Go
// definition. Any change to field order, type, or padding requires
// updating both sides and the matching layout tests.
//
// Copyright (C) 2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
// =============================================================================

#ifndef LUX_FHE_TYPES_CIPHERTEXT_HEADER_HPP
#define LUX_FHE_TYPES_CIPHERTEXT_HEADER_HPP

#include <cstddef>
#include <cstdint>

#include "lux/lattice/types/domain.hpp"

namespace lux::fhe {

// FHEScheme identifies the FHE scheme that produced a ciphertext.
// Wire-stable: enum values are fixed; add new schemes by appending.
enum class FHEScheme : std::uint32_t {
    TFHE = 0,
    FHEW = 1,
    CKKS = 2,
    BFV  = 3,
    BGV  = 4,
};

// FHECiphertextHeader wraps every FHE buffer with the metadata required for
// safe dispatch. Layout matches Go's FHECiphertextHeader exactly.
//
// Total size: 144 bytes.
struct FHECiphertextHeader {
    std::uint8_t            params_hash[32];   // offset 0,   size 32
    std::uint8_t            key_id[32];        // offset 32,  size 32
    std::uint8_t            circuit_id[32];    // offset 64,  size 32
    FHEScheme               scheme;            // offset 96,  size 4
    std::uint32_t           level;             // offset 100, size 4
    std::uint32_t           N;                 // offset 104, size 4
    std::uint32_t           modulus_count;     // offset 108, size 4
    lux::lattice::PolyDomain domain;           // offset 112, size 1
    std::uint8_t            _pad[7];           // offset 113, size 7
    std::uint8_t            reserved[24];      // offset 120, size 24
};

static_assert(sizeof(FHECiphertextHeader) == 144,
              "FHECiphertextHeader must be 144 bytes");
static_assert(offsetof(FHECiphertextHeader, params_hash)   == 0,
              "params_hash offset");
static_assert(offsetof(FHECiphertextHeader, key_id)        == 32,
              "key_id offset");
static_assert(offsetof(FHECiphertextHeader, circuit_id)    == 64,
              "circuit_id offset");
static_assert(offsetof(FHECiphertextHeader, scheme)        == 96,
              "scheme offset");
static_assert(offsetof(FHECiphertextHeader, level)         == 100,
              "level offset");
static_assert(offsetof(FHECiphertextHeader, N)             == 104,
              "N offset");
static_assert(offsetof(FHECiphertextHeader, modulus_count) == 108,
              "modulus_count offset");
static_assert(offsetof(FHECiphertextHeader, domain)        == 112,
              "domain offset");
static_assert(offsetof(FHECiphertextHeader, reserved)      == 120,
              "reserved offset");

} // namespace lux::fhe

#endif // LUX_FHE_TYPES_CIPHERTEXT_HEADER_HPP
