// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// Benchmark suite for OpenFHE PKE operations (CKKS, BGV, BFV)

#include <openfhe.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace lux::fhe;
using namespace std::chrono;

template<typename F>
double benchmark(F&& func, int iterations = 50) {
    for (int i = 0; i < 3; i++) func();
    
    std::vector<double> times;
    for (int i = 0; i < iterations; i++) {
        auto start = high_resolution_clock::now();
        func();
        auto end = high_resolution_clock::now();
        times.push_back(duration_cast<microseconds>(end - start).count() / 1000.0);
    }
    
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

void benchmark_ckks() {
    std::cout << "\n=== CKKS Benchmarks ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    
    // Setup parameters
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(5);
    params.SetScalingModSize(50);
    params.SetBatchSize(8192);
    params.SetSecurityLevel(HEStd_128_classic);
    
    auto start = high_resolution_clock::now();
    auto cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    auto setupTime = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
    std::cout << "Context setup: " << setupTime << " ms" << std::endl;
    
    // Key generation
    start = high_resolution_clock::now();
    auto keys = cc->KeyGen();
    auto keygenTime = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
    std::cout << "Key generation: " << keygenTime << " ms" << std::endl;
    
    // Eval key generation
    start = high_resolution_clock::now();
    cc->EvalMultKeyGen(keys.secretKey);
    auto evalKeyTime = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
    std::cout << "Eval mult key gen: " << evalKeyTime << " ms" << std::endl;
    
    // Test vectors
    std::vector<double> x1 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    std::vector<double> x2 = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5};
    
    auto pt1 = cc->MakeCKKSPackedPlaintext(x1);
    auto pt2 = cc->MakeCKKSPackedPlaintext(x2);
    
    // Encryption
    auto ct1 = cc->Encrypt(keys.publicKey, pt1);
    auto ct2 = cc->Encrypt(keys.publicKey, pt2);
    
    double encryptTime = benchmark([&]() { cc->Encrypt(keys.publicKey, pt1); }, 50);
    std::cout << "Encrypt: " << encryptTime << " ms" << std::endl;
    
    // Decryption
    Plaintext result;
    double decryptTime = benchmark([&]() { cc->Decrypt(keys.secretKey, ct1, &result); }, 50);
    std::cout << "Decrypt: " << decryptTime << " ms" << std::endl;
    
    // Operations
    std::cout << "\n--- Homomorphic Operations ---" << std::endl;
    
    double addTime = benchmark([&]() { cc->EvalAdd(ct1, ct2); }, 100);
    std::cout << "Add: " << addTime << " ms" << std::endl;
    
    double subTime = benchmark([&]() { cc->EvalSub(ct1, ct2); }, 100);
    std::cout << "Sub: " << subTime << " ms" << std::endl;
    
    double mulTime = benchmark([&]() { cc->EvalMult(ct1, ct2); }, 50);
    std::cout << "Mult: " << mulTime << " ms" << std::endl;
    
    double scalarMulTime = benchmark([&]() { cc->EvalMult(ct1, 2.5); }, 100);
    std::cout << "Scalar mult: " << scalarMulTime << " ms" << std::endl;
    
    // Rotation (requires rotation key)
    cc->EvalRotateKeyGen(keys.secretKey, {1, 2, -1, -2});
    double rotateTime = benchmark([&]() { cc->EvalRotate(ct1, 1); }, 50);
    std::cout << "Rotate: " << rotateTime << " ms" << std::endl;
}

void benchmark_bgv() {
    std::cout << "\n=== BGV Benchmarks ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    
    CCParams<CryptoContextBGVRNS> params;
    params.SetMultiplicativeDepth(5);
    params.SetPlaintextModulus(65537);
    params.SetSecurityLevel(HEStd_128_classic);
    
    auto start = high_resolution_clock::now();
    auto cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    auto setupTime = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
    std::cout << "Context setup: " << setupTime << " ms" << std::endl;
    
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    
    std::vector<int64_t> x1 = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int64_t> x2 = {10, 20, 30, 40, 50, 60, 70, 80};
    
    auto pt1 = cc->MakePackedPlaintext(x1);
    auto pt2 = cc->MakePackedPlaintext(x2);
    
    auto ct1 = cc->Encrypt(keys.publicKey, pt1);
    auto ct2 = cc->Encrypt(keys.publicKey, pt2);
    
    double encryptTime = benchmark([&]() { cc->Encrypt(keys.publicKey, pt1); }, 50);
    std::cout << "Encrypt: " << encryptTime << " ms" << std::endl;
    
    Plaintext result;
    double decryptTime = benchmark([&]() { cc->Decrypt(keys.secretKey, ct1, &result); }, 50);
    std::cout << "Decrypt: " << decryptTime << " ms" << std::endl;
    
    std::cout << "\n--- Homomorphic Operations ---" << std::endl;
    
    double addTime = benchmark([&]() { cc->EvalAdd(ct1, ct2); }, 100);
    std::cout << "Add: " << addTime << " ms" << std::endl;
    
    double mulTime = benchmark([&]() { cc->EvalMult(ct1, ct2); }, 50);
    std::cout << "Mult: " << mulTime << " ms" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "OpenFHE PKE Benchmark Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    
    benchmark_ckks();
    benchmark_bgv();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmark complete" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
