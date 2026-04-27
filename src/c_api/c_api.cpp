// c_api.cpp - C API implementation for Lux FHE Library
//
// Copyright (c) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco

#include "lux/fhe/c_api.h"

#include "binfhecontext.h"
#include "utils/serial.h"
#include <cstring>
#include <sstream>
#include <vector>

using namespace lux::fhe;

// =============================================================================
// Internal Structures
// =============================================================================

struct LuxFheContext {
    BinFHEContext cc;
    BINFHE_PARAMSET paramSet;
    BINFHE_METHOD method;
};

struct LuxFheSecretKey {
    LWEPrivateKey sk;
};

struct LuxFhePublicKey {
    LWEPublicKey pk;
};

struct LuxFheBootstrapKey {
    bool generated = false;
};

struct LuxFheCiphertext {
    LWECiphertext ct;
};

// =============================================================================
// Helpers
// =============================================================================

static BINFHE_PARAMSET to_paramset(LuxFheParams p) {
    switch (p) {
        case LUX_FHE_PARAMS_TOY:     return TOY;
        case LUX_FHE_PARAMS_MEDIUM:  return MEDIUM;
        case LUX_FHE_PARAMS_STD128:  return STD128_LMKCDEY;
        case LUX_FHE_PARAMS_STD128Q: return STD128Q_LMKCDEY;
        case LUX_FHE_PARAMS_STD192:  return STD192_LMKCDEY;
        case LUX_FHE_PARAMS_STD192Q: return STD192Q_LMKCDEY;
        case LUX_FHE_PARAMS_STD256:  return STD256_LMKCDEY;
        case LUX_FHE_PARAMS_STD256Q: return STD256Q_LMKCDEY;
        default:                     return STD128_LMKCDEY;
    }
}

static BINFHE_METHOD to_method(LuxFheMethod m) {
    switch (m) {
        case LUX_FHE_METHOD_AP:      return AP;
        case LUX_FHE_METHOD_GINX:    return GINX;
        case LUX_FHE_METHOD_LMKCDEY: return LMKCDEY;
        default:                     return LMKCDEY;
    }
}

// =============================================================================
// Context
// =============================================================================

LuxFheError lux_fhe_context_new(LuxFheContext** ctx, LuxFheParams params, LuxFheMethod method) {
    if (!ctx) return LUX_FHE_ERR_NULL_PTR;
    try {
        auto* c = new LuxFheContext();
        c->paramSet = to_paramset(params);
        c->method = to_method(method);
        c->cc.GenerateBinFHEContext(c->paramSet, c->method);
        *ctx = c;
        return LUX_FHE_OK;
    } catch (...) {
        return LUX_FHE_ERR_ALLOC;
    }
}

void lux_fhe_context_free(LuxFheContext* ctx) { delete ctx; }

uint32_t lux_fhe_context_n(const LuxFheContext* ctx) {
    if (!ctx) return 0;
    // Cast away const - GetParams() should be const but isn't marked as such
    auto* nc = const_cast<LuxFheContext*>(ctx);
    return nc->cc.GetParams()->GetLWEParams()->Getn();
}

uint32_t lux_fhe_context_ring_dim(const LuxFheContext* ctx) {
    if (!ctx) return 0;
    auto* nc = const_cast<LuxFheContext*>(ctx);
    return nc->cc.GetParams()->GetRingGSWParams()->GetN();
}

uint64_t lux_fhe_context_modulus(const LuxFheContext* ctx) {
    if (!ctx) return 0;
    auto* nc = const_cast<LuxFheContext*>(ctx);
    auto q = nc->cc.GetParams()->GetLWEParams()->Getq();
    return static_cast<uint64_t>(q.ConvertToInt());
}

// =============================================================================
// Key Generation
// =============================================================================

LuxFheError lux_fhe_keygen_secret(LuxFheContext* ctx, LuxFheSecretKey** sk) {
    if (!ctx || !sk) return LUX_FHE_ERR_NULL_PTR;
    try {
        auto* k = new LuxFheSecretKey();
        k->sk = ctx->cc.KeyGen();
        *sk = k;
        return LUX_FHE_OK;
    } catch (...) {
        return LUX_FHE_ERR_KEYGEN;
    }
}

LuxFheError lux_fhe_keygen_public(LuxFheContext* ctx, const LuxFheSecretKey* sk, LuxFhePublicKey** pk) {
    if (!ctx || !sk || !pk) return LUX_FHE_ERR_NULL_PTR;
    try {
        auto* k = new LuxFhePublicKey();
        k->pk = ctx->cc.PubKeyGen(sk->sk);
        *pk = k;
        return LUX_FHE_OK;
    } catch (...) {
        return LUX_FHE_ERR_KEYGEN;
    }
}

LuxFheError lux_fhe_keygen_bootstrap(LuxFheContext* ctx, const LuxFheSecretKey* sk, LuxFheBootstrapKey** bsk) {
    if (!ctx || !sk || !bsk) return LUX_FHE_ERR_NULL_PTR;
    try {
        ctx->cc.BTKeyGen(sk->sk);
        auto* k = new LuxFheBootstrapKey();
        k->generated = true;
        *bsk = k;
        return LUX_FHE_OK;
    } catch (...) {
        return LUX_FHE_ERR_KEYGEN;
    }
}

void lux_fhe_secretkey_free(LuxFheSecretKey* sk) { delete sk; }
void lux_fhe_publickey_free(LuxFhePublicKey* pk) { delete pk; }
void lux_fhe_bootstrapkey_free(LuxFheBootstrapKey* bsk) { delete bsk; }

// =============================================================================
// Encryption / Decryption
// =============================================================================

LuxFheError lux_fhe_encrypt(LuxFheContext* ctx, const LuxFheSecretKey* sk, bool plaintext, LuxFheCiphertext** ct) {
    if (!ctx || !sk || !ct) return LUX_FHE_ERR_NULL_PTR;
    try {
        auto* c = new LuxFheCiphertext();
        c->ct = ctx->cc.Encrypt(sk->sk, plaintext ? 1 : 0, FRESH, 4);
        *ct = c;
        return LUX_FHE_OK;
    } catch (...) {
        return LUX_FHE_ERR_ENCRYPT;
    }
}

LuxFheError lux_fhe_encrypt_pk(LuxFheContext* ctx, const LuxFhePublicKey* pk, bool plaintext, LuxFheCiphertext** ct) {
    if (!ctx || !pk || !ct) return LUX_FHE_ERR_NULL_PTR;
    try {
        auto* c = new LuxFheCiphertext();
        c->ct = ctx->cc.Encrypt(pk->pk, plaintext ? 1 : 0, FRESH, 4);
        *ct = c;
        return LUX_FHE_OK;
    } catch (...) {
        return LUX_FHE_ERR_ENCRYPT;
    }
}

LuxFheError lux_fhe_decrypt(LuxFheContext* ctx, const LuxFheSecretKey* sk, const LuxFheCiphertext* ct, bool* plaintext) {
    if (!ctx || !sk || !ct || !plaintext) return LUX_FHE_ERR_NULL_PTR;
    try {
        LWEPlaintext result;
        ctx->cc.Decrypt(sk->sk, ct->ct, &result, 4);
        *plaintext = (result == 1);
        return LUX_FHE_OK;
    } catch (...) {
        return LUX_FHE_ERR_DECRYPT;
    }
}

void lux_fhe_ciphertext_free(LuxFheCiphertext* ct) { delete ct; }

LuxFheError lux_fhe_ciphertext_clone(const LuxFheCiphertext* src, LuxFheCiphertext** dst) {
    if (!src || !dst) return LUX_FHE_ERR_NULL_PTR;
    try {
        auto* c = new LuxFheCiphertext();
        c->ct = src->ct;
        *dst = c;
        return LUX_FHE_OK;
    } catch (...) {
        return LUX_FHE_ERR_ALLOC;
    }
}

// =============================================================================
// Gates
// =============================================================================

LuxFheError lux_fhe_not(LuxFheContext* ctx, const LuxFheCiphertext* ct, LuxFheCiphertext** result) {
    if (!ctx || !ct || !result) return LUX_FHE_ERR_NULL_PTR;
    try {
        auto* r = new LuxFheCiphertext();
        r->ct = ctx->cc.EvalNOT(ct->ct);
        *result = r;
        return LUX_FHE_OK;
    } catch (...) {
        return LUX_FHE_ERR_GATE;
    }
}

#define IMPL_GATE(name, op) \
LuxFheError lux_fhe_##name(LuxFheContext* ctx, const LuxFheBootstrapKey* bsk, \
    const LuxFheCiphertext* a, const LuxFheCiphertext* b, LuxFheCiphertext** result) { \
    if (!ctx || !bsk || !a || !b || !result) return LUX_FHE_ERR_NULL_PTR; \
    if (!bsk->generated) return LUX_FHE_ERR_NOT_INIT; \
    try { \
        auto* r = new LuxFheCiphertext(); \
        r->ct = ctx->cc.EvalBinGate(op, a->ct, b->ct); \
        *result = r; \
        return LUX_FHE_OK; \
    } catch (...) { return LUX_FHE_ERR_GATE; } \
}

IMPL_GATE(and, AND)
IMPL_GATE(or, OR)
IMPL_GATE(xor, XOR)
IMPL_GATE(nand, NAND)
IMPL_GATE(nor, NOR)
IMPL_GATE(xnor, XNOR)

LuxFheError lux_fhe_mux(LuxFheContext* ctx, const LuxFheBootstrapKey* bsk,
    const LuxFheCiphertext* sel, const LuxFheCiphertext* a, const LuxFheCiphertext* b, LuxFheCiphertext** result) {
    if (!ctx || !bsk || !sel || !a || !b || !result) return LUX_FHE_ERR_NULL_PTR;
    if (!bsk->generated) return LUX_FHE_ERR_NOT_INIT;
    try {
        auto* r = new LuxFheCiphertext();
        // CMUX takes 3 inputs as a vector: {selector, true_value, false_value}
        std::vector<LWECiphertext> inputs = {sel->ct, a->ct, b->ct};
        r->ct = ctx->cc.EvalBinGate(CMUX, inputs);
        *result = r;
        return LUX_FHE_OK;
    } catch (...) {
        return LUX_FHE_ERR_GATE;
    }
}

// =============================================================================
// Bootstrap
// =============================================================================

LuxFheError lux_fhe_bootstrap(LuxFheContext* ctx, const LuxFheBootstrapKey* bsk,
    const LuxFheCiphertext* ct, LuxFheCiphertext** result) {
    if (!ctx || !bsk || !ct || !result) return LUX_FHE_ERR_NULL_PTR;
    if (!bsk->generated) return LUX_FHE_ERR_NOT_INIT;
    try {
        auto* r = new LuxFheCiphertext();
        r->ct = ctx->cc.Bootstrap(ct->ct);
        *result = r;
        return LUX_FHE_OK;
    } catch (...) {
        return LUX_FHE_ERR_BOOTSTRAP;
    }
}

// =============================================================================
// Serialization
// =============================================================================

LuxFheError lux_fhe_secretkey_marshal(const LuxFheSecretKey* sk, uint8_t** data, size_t* len) {
    if (!sk || !data || !len) return LUX_FHE_ERR_NULL_PTR;
    try {
        std::stringstream ss;
        Serial::Serialize(sk->sk, ss, SerType::BINARY);
        auto s = ss.str();
        *len = s.size();
        *data = new uint8_t[*len];
        std::memcpy(*data, s.data(), *len);
        return LUX_FHE_OK;
    } catch (...) {
        return LUX_FHE_ERR_SERIALIZE;
    }
}

LuxFheError lux_fhe_secretkey_unmarshal(LuxFheContext* ctx, const uint8_t* data, size_t len, LuxFheSecretKey** sk) {
    if (!ctx || !data || !sk) return LUX_FHE_ERR_NULL_PTR;
    try {
        std::stringstream ss;
        ss.write(reinterpret_cast<const char*>(data), len);
        auto* k = new LuxFheSecretKey();
        Serial::Deserialize(k->sk, ss, SerType::BINARY);
        *sk = k;
        return LUX_FHE_OK;
    } catch (...) {
        return LUX_FHE_ERR_DESERIALIZE;
    }
}

LuxFheError lux_fhe_ciphertext_marshal(const LuxFheCiphertext* ct, uint8_t** data, size_t* len) {
    if (!ct || !data || !len) return LUX_FHE_ERR_NULL_PTR;
    try {
        std::stringstream ss;
        Serial::Serialize(ct->ct, ss, SerType::BINARY);
        auto s = ss.str();
        *len = s.size();
        *data = new uint8_t[*len];
        std::memcpy(*data, s.data(), *len);
        return LUX_FHE_OK;
    } catch (...) {
        return LUX_FHE_ERR_SERIALIZE;
    }
}

LuxFheError lux_fhe_ciphertext_unmarshal(LuxFheContext* ctx, const uint8_t* data, size_t len, LuxFheCiphertext** ct) {
    if (!ctx || !data || !ct) return LUX_FHE_ERR_NULL_PTR;
    try {
        std::stringstream ss;
        ss.write(reinterpret_cast<const char*>(data), len);
        auto* c = new LuxFheCiphertext();
        Serial::Deserialize(c->ct, ss, SerType::BINARY);
        *ct = c;
        return LUX_FHE_OK;
    } catch (...) {
        return LUX_FHE_ERR_DESERIALIZE;
    }
}

void lux_fhe_bytes_free(uint8_t* data) { delete[] data; }

// =============================================================================
// Utility
// =============================================================================

const char* lux_fhe_version(void) { return "1.4.2"; }

const char* lux_fhe_strerror(LuxFheError err) {
    switch (err) {
        case LUX_FHE_OK:              return "ok";
        case LUX_FHE_ERR_NULL_PTR:    return "null pointer";
        case LUX_FHE_ERR_INVALID_PARAM: return "invalid parameter";
        case LUX_FHE_ERR_ALLOC:       return "allocation failed";
        case LUX_FHE_ERR_KEYGEN:      return "key generation failed";
        case LUX_FHE_ERR_ENCRYPT:     return "encryption failed";
        case LUX_FHE_ERR_DECRYPT:     return "decryption failed";
        case LUX_FHE_ERR_BOOTSTRAP:   return "bootstrap failed";
        case LUX_FHE_ERR_GATE:        return "gate evaluation failed";
        case LUX_FHE_ERR_SERIALIZE:   return "serialization failed";
        case LUX_FHE_ERR_DESERIALIZE: return "deserialization failed";
        case LUX_FHE_ERR_NOT_INIT:    return "not initialized";
        default:                       return "unknown error";
    }
}

bool lux_fhe_has_gpu(void) {
#if defined(WITH_GPU) || defined(WITH_MLX) || defined(WITH_CUDA)
    return true;
#else
    return false;
#endif
}
