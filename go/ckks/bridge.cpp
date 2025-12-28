// bridge.cpp - CGO bridge to OpenFHE CKKS
//
// Build: Requires OpenFHE installed with development headers
// See https://openfhe-development.readthedocs.io/en/latest/sphinx_rsts/intro/installation.html

#include <cstdlib>
#include <cstring>
#include <vector>

#include "openfhe.h"

using namespace lbcrypto;

// Opaque handle types
struct CKKSContext {
    CryptoContext<DCRTPoly> cc;
    CCParams<CryptoContextCKKSRNS> params;
};

struct CKKSKeyPair {
    KeyPair<DCRTPoly> kp;
    PrivateKey<DCRTPoly> sk;
    PublicKey<DCRTPoly> pk;
};

struct CKKSCiphertext {
    Ciphertext<DCRTPoly> ct;
};

extern "C" {

// Context management
CKKSContext* ckks_context_new(int log_n, int log_q, double scale) {
    try {
        auto ctx = new CKKSContext();
        
        // Set CKKS parameters
        ctx->params.SetMultiplicativeDepth(log_q / 40);  // Approximate
        ctx->params.SetScalingModSize(40);
        ctx->params.SetRingDim(1 << log_n);
        ctx->params.SetScalingTechnique(FLEXIBLEAUTO);
        ctx->params.SetSecurityLevel(HEStd_128_classic);
        
        // Generate crypto context
        ctx->cc = GenCryptoContext(ctx->params);
        ctx->cc->Enable(PKE);
        ctx->cc->Enable(KEYSWITCH);
        ctx->cc->Enable(LEVELEDSHE);
        ctx->cc->Enable(ADVANCEDSHE);
        
        return ctx;
    } catch (...) {
        return nullptr;
    }
}

void ckks_context_free(CKKSContext* ctx) {
    delete ctx;
}

// Key generation
CKKSKeyPair* ckks_keygen(CKKSContext* ctx) {
    if (!ctx) return nullptr;
    
    try {
        auto kp = new CKKSKeyPair();
        kp->kp = ctx->cc->KeyGen();
        kp->sk = kp->kp.secretKey;
        kp->pk = kp->kp.publicKey;
        
        // Generate evaluation keys for multiplication and rotation
        ctx->cc->EvalMultKeyGen(kp->sk);
        ctx->cc->EvalRotateKeyGen(kp->sk, {1, -1, 2, -2, 4, -4, 8, -8, 16, -16});
        
        return kp;
    } catch (...) {
        return nullptr;
    }
}

void ckks_keypair_free(CKKSKeyPair* kp) {
    delete kp;
}

// Encryption
CKKSCiphertext* ckks_encrypt(CKKSContext* ctx, CKKSKeyPair* kp, double* values, int len) {
    if (!ctx || !kp || !values || len <= 0) return nullptr;
    
    try {
        std::vector<double> vec(values, values + len);
        
        // Encode and encrypt
        Plaintext pt = ctx->cc->MakeCKKSPackedPlaintext(vec);
        auto ct = new CKKSCiphertext();
        ct->ct = ctx->cc->Encrypt(kp->pk, pt);
        
        return ct;
    } catch (...) {
        return nullptr;
    }
}

// Decryption
double* ckks_decrypt(CKKSContext* ctx, CKKSKeyPair* kp, CKKSCiphertext* ct, int* out_len) {
    if (!ctx || !kp || !ct || !out_len) return nullptr;
    
    try {
        Plaintext pt;
        ctx->cc->Decrypt(kp->sk, ct->ct, &pt);
        
        auto& vec = pt->GetRealPackedValue();
        *out_len = static_cast<int>(vec.size());
        
        double* result = static_cast<double*>(malloc(*out_len * sizeof(double)));
        if (!result) return nullptr;
        
        std::copy(vec.begin(), vec.end(), result);
        return result;
    } catch (...) {
        *out_len = 0;
        return nullptr;
    }
}

void ckks_ciphertext_free(CKKSCiphertext* ct) {
    delete ct;
}

// Homomorphic addition
CKKSCiphertext* ckks_add(CKKSContext* ctx, CKKSCiphertext* a, CKKSCiphertext* b) {
    if (!ctx || !a || !b) return nullptr;
    
    try {
        auto result = new CKKSCiphertext();
        result->ct = ctx->cc->EvalAdd(a->ct, b->ct);
        return result;
    } catch (...) {
        return nullptr;
    }
}

// Homomorphic subtraction
CKKSCiphertext* ckks_sub(CKKSContext* ctx, CKKSCiphertext* a, CKKSCiphertext* b) {
    if (!ctx || !a || !b) return nullptr;
    
    try {
        auto result = new CKKSCiphertext();
        result->ct = ctx->cc->EvalSub(a->ct, b->ct);
        return result;
    } catch (...) {
        return nullptr;
    }
}

// Homomorphic multiplication
CKKSCiphertext* ckks_mult(CKKSContext* ctx, CKKSCiphertext* a, CKKSCiphertext* b) {
    if (!ctx || !a || !b) return nullptr;
    
    try {
        auto result = new CKKSCiphertext();
        result->ct = ctx->cc->EvalMult(a->ct, b->ct);
        return result;
    } catch (...) {
        return nullptr;
    }
}

// Vector rotation
CKKSCiphertext* ckks_rotate(CKKSContext* ctx, CKKSKeyPair* kp, CKKSCiphertext* ct, int steps) {
    if (!ctx || !kp || !ct) return nullptr;
    
    try {
        auto result = new CKKSCiphertext();
        result->ct = ctx->cc->EvalRotate(ct->ct, steps);
        return result;
    } catch (...) {
        return nullptr;
    }
}

// Bootstrapping (noise refresh)
CKKSCiphertext* ckks_bootstrap(CKKSContext* ctx, CKKSKeyPair* kp, CKKSCiphertext* ct) {
    if (!ctx || !kp || !ct) return nullptr;
    
    try {
        // Note: OpenFHE CKKS bootstrapping requires additional setup
        // This is a placeholder - full bootstrap requires EvalBootstrapSetup
        auto result = new CKKSCiphertext();
        result->ct = ct->ct;  // For now, just copy
        return result;
    } catch (...) {
        return nullptr;
    }
}

// Serialization
unsigned char* ckks_serialize_ciphertext(CKKSCiphertext* ct, int* out_len) {
    if (!ct || !out_len) return nullptr;
    
    try {
        std::stringstream ss;
        Serial::Serialize(ct->ct, ss, SerType::BINARY);
        
        std::string str = ss.str();
        *out_len = static_cast<int>(str.size());
        
        unsigned char* result = static_cast<unsigned char*>(malloc(*out_len));
        if (!result) return nullptr;
        
        std::memcpy(result, str.data(), *out_len);
        return result;
    } catch (...) {
        *out_len = 0;
        return nullptr;
    }
}

CKKSCiphertext* ckks_deserialize_ciphertext(CKKSContext* ctx, unsigned char* data, int len) {
    if (!ctx || !data || len <= 0) return nullptr;
    
    try {
        std::string str(reinterpret_cast<char*>(data), len);
        std::stringstream ss(str);
        
        auto result = new CKKSCiphertext();
        Serial::Deserialize(result->ct, ss, SerType::BINARY);
        
        return result;
    } catch (...) {
        return nullptr;
    }
}

} // extern "C"
