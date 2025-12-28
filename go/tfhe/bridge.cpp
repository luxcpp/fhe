// bridge.cpp - C bridge for OpenFHE TFHE bindings
//
// This file provides C-compatible wrappers around the OpenFHE C++ API
// for use with Go's CGO.

#include <binfhe/binfhecontext.h>
#include <cstdlib>

using namespace lbcrypto;

extern "C" {

// Context management

void* NewBinFHEContext() {
    auto* ctx = new BinFHEContext();
    return static_cast<void*>(ctx);
}

void FreeBinFHEContext(void* ctx) {
    delete static_cast<BinFHEContext*>(ctx);
}

void GenerateBinFHEContext(void* ctx, int securityLevel, int method) {
    auto* cc = static_cast<BinFHEContext*>(ctx);
    
    BINFHE_PARAMSET security;
    switch (securityLevel) {
        case 0: security = STD128; break;
        case 1: security = STD192; break;
        case 2: security = STD256; break;
        default: security = STD128;
    }
    
    BINFHE_METHOD meth;
    switch (method) {
        case 0: meth = GINX; break;
        case 1: meth = AP; break;
        case 2: meth = LMKCDEY; break;
        default: meth = GINX;
    }
    
    cc->GenerateBinFHEContext(security, meth);
}

// Key management

void* KeyGen(void* ctx) {
    auto* cc = static_cast<BinFHEContext*>(ctx);
    auto sk = cc->KeyGen();
    auto* skPtr = new LWEPrivateKey(sk);
    return static_cast<void*>(skPtr);
}

void FreeLWESecretKey(void* sk) {
    delete static_cast<LWEPrivateKey*>(sk);
}

void BTKeyGen(void* ctx, void* sk) {
    auto* cc = static_cast<BinFHEContext*>(ctx);
    auto* skPtr = static_cast<LWEPrivateKey*>(sk);
    cc->BTKeyGen(*skPtr);
}

// Encryption/Decryption

void* Encrypt(void* ctx, void* sk, int plaintext) {
    auto* cc = static_cast<BinFHEContext*>(ctx);
    auto* skPtr = static_cast<LWEPrivateKey*>(sk);
    auto ct = cc->Encrypt(*skPtr, plaintext);
    auto* ctPtr = new LWECiphertext(ct);
    return static_cast<void*>(ctPtr);
}

int Decrypt(void* ctx, void* sk, void* ct) {
    auto* cc = static_cast<BinFHEContext*>(ctx);
    auto* skPtr = static_cast<LWEPrivateKey*>(sk);
    auto* ctPtr = static_cast<LWECiphertext*>(ct);
    LWEPlaintext result;
    cc->Decrypt(*skPtr, *ctPtr, &result);
    return static_cast<int>(result);
}

void FreeLWECiphertext(void* ct) {
    delete static_cast<LWECiphertext*>(ct);
}

// Gate operations

void* EvalAND(void* ctx, void* ct1, void* ct2) {
    auto* cc = static_cast<BinFHEContext*>(ctx);
    auto* ct1Ptr = static_cast<LWECiphertext*>(ct1);
    auto* ct2Ptr = static_cast<LWECiphertext*>(ct2);
    auto result = cc->EvalBinGate(AND, *ct1Ptr, *ct2Ptr);
    return static_cast<void*>(new LWECiphertext(result));
}

void* EvalOR(void* ctx, void* ct1, void* ct2) {
    auto* cc = static_cast<BinFHEContext*>(ctx);
    auto* ct1Ptr = static_cast<LWECiphertext*>(ct1);
    auto* ct2Ptr = static_cast<LWECiphertext*>(ct2);
    auto result = cc->EvalBinGate(OR, *ct1Ptr, *ct2Ptr);
    return static_cast<void*>(new LWECiphertext(result));
}

void* EvalXOR(void* ctx, void* ct1, void* ct2) {
    auto* cc = static_cast<BinFHEContext*>(ctx);
    auto* ct1Ptr = static_cast<LWECiphertext*>(ct1);
    auto* ct2Ptr = static_cast<LWECiphertext*>(ct2);
    auto result = cc->EvalBinGate(XOR, *ct1Ptr, *ct2Ptr);
    return static_cast<void*>(new LWECiphertext(result));
}

void* EvalNOT(void* ctx, void* ct) {
    auto* cc = static_cast<BinFHEContext*>(ctx);
    auto* ctPtr = static_cast<LWECiphertext*>(ct);
    auto result = cc->EvalNOT(*ctPtr);
    return static_cast<void*>(new LWECiphertext(result));
}

void* EvalNAND(void* ctx, void* ct1, void* ct2) {
    auto* cc = static_cast<BinFHEContext*>(ctx);
    auto* ct1Ptr = static_cast<LWECiphertext*>(ct1);
    auto* ct2Ptr = static_cast<LWECiphertext*>(ct2);
    auto result = cc->EvalBinGate(NAND, *ct1Ptr, *ct2Ptr);
    return static_cast<void*>(new LWECiphertext(result));
}

void* EvalNOR(void* ctx, void* ct1, void* ct2) {
    auto* cc = static_cast<BinFHEContext*>(ctx);
    auto* ct1Ptr = static_cast<LWECiphertext*>(ct1);
    auto* ct2Ptr = static_cast<LWECiphertext*>(ct2);
    auto result = cc->EvalBinGate(NOR, *ct1Ptr, *ct2Ptr);
    return static_cast<void*>(new LWECiphertext(result));
}

void* EvalXNOR(void* ctx, void* ct1, void* ct2) {
    auto* cc = static_cast<BinFHEContext*>(ctx);
    auto* ct1Ptr = static_cast<LWECiphertext*>(ct1);
    auto* ct2Ptr = static_cast<LWECiphertext*>(ct2);
    auto result = cc->EvalBinGate(XNOR, *ct1Ptr, *ct2Ptr);
    return static_cast<void*>(new LWECiphertext(result));
}

} // extern "C"
