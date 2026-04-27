// c_api.h - C API for Lux FHE Library
//
// Provides C-compatible interface to the Lux FHE library for FFI consumers
// including Go CGO, Python ctypes, and other language bindings.
//
// Copyright (c) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// BSD-3-Clause-Eco License Terms:
// - Free for use on Lux Network primary network and testnets
// - Commercial licensing required for other networks
// - Contact: licensing@lux.network

#ifndef LUX_FHE_C_API_H
#define LUX_FHE_C_API_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Export Macros
// =============================================================================

#ifdef _WIN32
  #ifdef LUX_FHE_BUILDING
    #define LUX_FHE_API __declspec(dllexport)
  #else
    #define LUX_FHE_API __declspec(dllimport)
  #endif
#else
  #ifdef LUX_FHE_BUILDING
    #define LUX_FHE_API __attribute__((visibility("default")))
  #else
    #define LUX_FHE_API
  #endif
#endif

// =============================================================================
// Opaque Handle Types
// =============================================================================

typedef struct LuxFheContext LuxFheContext;
typedef struct LuxFheSecretKey LuxFheSecretKey;
typedef struct LuxFhePublicKey LuxFhePublicKey;
typedef struct LuxFheBootstrapKey LuxFheBootstrapKey;
typedef struct LuxFheCiphertext LuxFheCiphertext;

// =============================================================================
// Error Codes
// =============================================================================

typedef enum {
    LUX_FHE_OK = 0,
    LUX_FHE_ERR_NULL_PTR = -1,
    LUX_FHE_ERR_INVALID_PARAM = -2,
    LUX_FHE_ERR_ALLOC = -3,
    LUX_FHE_ERR_KEYGEN = -4,
    LUX_FHE_ERR_ENCRYPT = -5,
    LUX_FHE_ERR_DECRYPT = -6,
    LUX_FHE_ERR_BOOTSTRAP = -7,
    LUX_FHE_ERR_GATE = -8,
    LUX_FHE_ERR_SERIALIZE = -9,
    LUX_FHE_ERR_DESERIALIZE = -10,
    LUX_FHE_ERR_NOT_INIT = -11,
} LuxFheError;

// =============================================================================
// Parameter Sets
// =============================================================================

typedef enum {
    LUX_FHE_PARAMS_TOY = 0,          // Insecure, testing only
    LUX_FHE_PARAMS_MEDIUM = 1,       // Medium security
    LUX_FHE_PARAMS_STD128 = 2,       // 128-bit classical security
    LUX_FHE_PARAMS_STD128Q = 3,      // 128-bit post-quantum security
    LUX_FHE_PARAMS_STD192 = 4,       // 192-bit security
    LUX_FHE_PARAMS_STD192Q = 5,      // 192-bit post-quantum
    LUX_FHE_PARAMS_STD256 = 6,       // 256-bit security
    LUX_FHE_PARAMS_STD256Q = 7,      // 256-bit post-quantum
} LuxFheParams;

// =============================================================================
// Bootstrapping Methods
// =============================================================================

typedef enum {
    LUX_FHE_METHOD_AP = 0,       // Alperin-Pulver
    LUX_FHE_METHOD_GINX = 1,     // GINX
    LUX_FHE_METHOD_LMKCDEY = 2,  // LMKCDEY (fastest, default)
} LuxFheMethod;

// =============================================================================
// Context Management
// =============================================================================

// Create FHE context with parameter set
LUX_FHE_API LuxFheError lux_fhe_context_new(
    LuxFheContext** ctx,
    LuxFheParams params,
    LuxFheMethod method
);

// Free FHE context
LUX_FHE_API void lux_fhe_context_free(LuxFheContext* ctx);

// Get LWE dimension
LUX_FHE_API uint32_t lux_fhe_context_n(const LuxFheContext* ctx);

// Get ring dimension
LUX_FHE_API uint32_t lux_fhe_context_ring_dim(const LuxFheContext* ctx);

// Get modulus
LUX_FHE_API uint64_t lux_fhe_context_modulus(const LuxFheContext* ctx);

// =============================================================================
// Key Generation
// =============================================================================

// Generate secret key
LUX_FHE_API LuxFheError lux_fhe_keygen_secret(
    LuxFheContext* ctx,
    LuxFheSecretKey** sk
);

// Generate public key from secret key
LUX_FHE_API LuxFheError lux_fhe_keygen_public(
    LuxFheContext* ctx,
    const LuxFheSecretKey* sk,
    LuxFhePublicKey** pk
);

// Generate bootstrap key (enables homomorphic gates)
LUX_FHE_API LuxFheError lux_fhe_keygen_bootstrap(
    LuxFheContext* ctx,
    const LuxFheSecretKey* sk,
    LuxFheBootstrapKey** bsk
);

// Free keys
LUX_FHE_API void lux_fhe_secretkey_free(LuxFheSecretKey* sk);
LUX_FHE_API void lux_fhe_publickey_free(LuxFhePublicKey* pk);
LUX_FHE_API void lux_fhe_bootstrapkey_free(LuxFheBootstrapKey* bsk);

// =============================================================================
// Encryption / Decryption
// =============================================================================

// Encrypt bit with secret key
LUX_FHE_API LuxFheError lux_fhe_encrypt(
    LuxFheContext* ctx,
    const LuxFheSecretKey* sk,
    bool plaintext,
    LuxFheCiphertext** ct
);

// Encrypt bit with public key
LUX_FHE_API LuxFheError lux_fhe_encrypt_pk(
    LuxFheContext* ctx,
    const LuxFhePublicKey* pk,
    bool plaintext,
    LuxFheCiphertext** ct
);

// Decrypt ciphertext
LUX_FHE_API LuxFheError lux_fhe_decrypt(
    LuxFheContext* ctx,
    const LuxFheSecretKey* sk,
    const LuxFheCiphertext* ct,
    bool* plaintext
);

// Free ciphertext
LUX_FHE_API void lux_fhe_ciphertext_free(LuxFheCiphertext* ct);

// Clone ciphertext
LUX_FHE_API LuxFheError lux_fhe_ciphertext_clone(
    const LuxFheCiphertext* src,
    LuxFheCiphertext** dst
);

// =============================================================================
// Boolean Gates (with bootstrapping)
// =============================================================================

// NOT gate (no bootstrapping)
LUX_FHE_API LuxFheError lux_fhe_not(
    LuxFheContext* ctx,
    const LuxFheCiphertext* ct,
    LuxFheCiphertext** result
);

// AND gate
LUX_FHE_API LuxFheError lux_fhe_and(
    LuxFheContext* ctx,
    const LuxFheBootstrapKey* bsk,
    const LuxFheCiphertext* a,
    const LuxFheCiphertext* b,
    LuxFheCiphertext** result
);

// OR gate
LUX_FHE_API LuxFheError lux_fhe_or(
    LuxFheContext* ctx,
    const LuxFheBootstrapKey* bsk,
    const LuxFheCiphertext* a,
    const LuxFheCiphertext* b,
    LuxFheCiphertext** result
);

// XOR gate
LUX_FHE_API LuxFheError lux_fhe_xor(
    LuxFheContext* ctx,
    const LuxFheBootstrapKey* bsk,
    const LuxFheCiphertext* a,
    const LuxFheCiphertext* b,
    LuxFheCiphertext** result
);

// NAND gate
LUX_FHE_API LuxFheError lux_fhe_nand(
    LuxFheContext* ctx,
    const LuxFheBootstrapKey* bsk,
    const LuxFheCiphertext* a,
    const LuxFheCiphertext* b,
    LuxFheCiphertext** result
);

// NOR gate
LUX_FHE_API LuxFheError lux_fhe_nor(
    LuxFheContext* ctx,
    const LuxFheBootstrapKey* bsk,
    const LuxFheCiphertext* a,
    const LuxFheCiphertext* b,
    LuxFheCiphertext** result
);

// XNOR gate
LUX_FHE_API LuxFheError lux_fhe_xnor(
    LuxFheContext* ctx,
    const LuxFheBootstrapKey* bsk,
    const LuxFheCiphertext* a,
    const LuxFheCiphertext* b,
    LuxFheCiphertext** result
);

// MUX gate (if sel then a else b)
LUX_FHE_API LuxFheError lux_fhe_mux(
    LuxFheContext* ctx,
    const LuxFheBootstrapKey* bsk,
    const LuxFheCiphertext* sel,
    const LuxFheCiphertext* a,
    const LuxFheCiphertext* b,
    LuxFheCiphertext** result
);

// =============================================================================
// Bootstrapping
// =============================================================================

// Refresh ciphertext (reduce noise)
LUX_FHE_API LuxFheError lux_fhe_bootstrap(
    LuxFheContext* ctx,
    const LuxFheBootstrapKey* bsk,
    const LuxFheCiphertext* ct,
    LuxFheCiphertext** result
);

// =============================================================================
// Serialization
// =============================================================================

// Serialize secret key
LUX_FHE_API LuxFheError lux_fhe_secretkey_marshal(
    const LuxFheSecretKey* sk,
    uint8_t** data,
    size_t* len
);

// Deserialize secret key
LUX_FHE_API LuxFheError lux_fhe_secretkey_unmarshal(
    LuxFheContext* ctx,
    const uint8_t* data,
    size_t len,
    LuxFheSecretKey** sk
);

// Serialize ciphertext
LUX_FHE_API LuxFheError lux_fhe_ciphertext_marshal(
    const LuxFheCiphertext* ct,
    uint8_t** data,
    size_t* len
);

// Deserialize ciphertext
LUX_FHE_API LuxFheError lux_fhe_ciphertext_unmarshal(
    LuxFheContext* ctx,
    const uint8_t* data,
    size_t len,
    LuxFheCiphertext** ct
);

// Free serialized bytes
LUX_FHE_API void lux_fhe_bytes_free(uint8_t* data);

// =============================================================================
// Utility
// =============================================================================

// Get library version
LUX_FHE_API const char* lux_fhe_version(void);

// Get error message
LUX_FHE_API const char* lux_fhe_strerror(LuxFheError err);

// Check if GPU acceleration is available
LUX_FHE_API bool lux_fhe_has_gpu(void);

#ifdef __cplusplus
}
#endif

#endif // LUX_FHE_C_API_H
