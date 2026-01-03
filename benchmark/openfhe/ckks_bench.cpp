// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025 Lux Industries Inc
// CKKS Benchmark - Matching Lux Lattice Go benchmarks for 1:1 comparison

#include "openfhe.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace lux::fhe;
using namespace std::chrono;

// Timing helper
template<typename Func>
double time_op(const std::string& name, Func f, int iterations = 20) {
    // Warmup
    for (int i = 0; i < 3; i++) f();
    
    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        f();
    }
    auto end = high_resolution_clock::now();
    
    auto duration = duration_cast<microseconds>(end - start).count();
    double avg_us = static_cast<double>(duration) / iterations;
    double avg_ms = avg_us / 1000.0;
    
    std::cout << std::left << std::setw(30) << name 
              << std::right << std::setw(12) << std::fixed << std::setprecision(2) 
              << avg_ms << " ms" << std::endl;
    
    return avg_ms;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "OpenFHE CKKS Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Parameters matching Go lattice: logN=15 (min for HE standards), similar modulus chain
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecurityLevel(HEStd_128_classic);
    parameters.SetRingDim(1 << 15);  // LogN=15, N=32768 (min for 128-bit)
    parameters.SetMultiplicativeDepth(7);  // Similar to Qi=8
    parameters.SetScalingModSize(40);  // LogScale=40
    parameters.SetBatchSize(1 << 14);  // N/2 slots
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    
    std::cout << "\nParameters:" << std::endl;
    std::cout << "  Ring dimension: " << (1 << 15) << std::endl;
    std::cout << "  Multiplicative depth: 7" << std::endl;
    std::cout << "  Scaling mod size: 40 bits" << std::endl;
    std::cout << "  Batch size: " << (1 << 14) << std::endl;
    
    // Generate context
    std::cout << "\nGenerating crypto context..." << std::endl;
    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    
    // Generate keys
    std::cout << "Generating keys..." << std::endl;
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    // Generate rotation keys for inner product (all powers of 2 up to batchSize)
    std::vector<int32_t> rotations = {1, 2, -1, -2};
    for (int i = 4; i <= (1 << 14); i *= 2) {
        rotations.push_back(i);
        rotations.push_back(-i);
    }
    cc->EvalRotateKeyGen(keys.secretKey, rotations);
    
    // Create test vectors
    std::vector<double> vec1(1 << 14);
    std::vector<double> vec2(1 << 14);
    for (size_t i = 0; i < vec1.size(); i++) {
        vec1[i] = 0.5 + 0.1 * sin(i * 0.01);
        vec2[i] = 0.3 + 0.1 * cos(i * 0.01);
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Encoder Benchmarks" << std::endl;
    std::cout << "========================================" << std::endl;
    
    Plaintext pt1, pt2;
    time_op("Encode", [&]() {
        pt1 = cc->MakeCKKSPackedPlaintext(vec1);
    });
    
    pt1 = cc->MakeCKKSPackedPlaintext(vec1);
    pt2 = cc->MakeCKKSPackedPlaintext(vec2);
    
    // Encrypt
    Ciphertext<DCRTPoly> ct1, ct2;
    std::cout << "\n========================================" << std::endl;
    std::cout << "Encryption/Decryption" << std::endl;
    std::cout << "========================================" << std::endl;
    
    time_op("Encrypt", [&]() {
        ct1 = cc->Encrypt(keys.publicKey, pt1);
    });
    
    ct1 = cc->Encrypt(keys.publicKey, pt1);
    ct2 = cc->Encrypt(keys.publicKey, pt2);
    
    Plaintext result_pt;
    time_op("Decrypt", [&]() {
        cc->Decrypt(keys.secretKey, ct1, &result_pt);
    });
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Evaluator Benchmarks (Single-threaded)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Add operations
    double scalar = 2.5;
    Ciphertext<DCRTPoly> result;
    
    time_op("Add/Scalar", [&]() {
        result = cc->EvalAdd(ct1, scalar);
    });
    
    time_op("Add/Plaintext", [&]() {
        result = cc->EvalAdd(ct1, pt2);
    });
    
    time_op("Add/Ciphertext", [&]() {
        result = cc->EvalAdd(ct1, ct2);
    });
    
    // Mul operations
    time_op("Mul/Scalar", [&]() {
        result = cc->EvalMult(ct1, scalar);
    });
    
    time_op("Mul/Plaintext", [&]() {
        result = cc->EvalMult(ct1, pt2);
    });
    
    time_op("Mul/Ciphertext", [&]() {
        result = cc->EvalMult(ct1, ct2);
    });
    
    // MulRelin (multiplication with relinearization)
    time_op("MulRelin/Ciphertext", [&]() {
        auto temp = cc->EvalMult(ct1, ct2);
        result = cc->Relinearize(temp);
    });
    
    // Rescale
    auto ct_mul = cc->EvalMult(ct1, ct2);
    time_op("Rescale", [&]() {
        result = cc->Rescale(ct_mul);
    });
    
    // Rotation
    time_op("Rotate (by 1)", [&]() {
        result = cc->EvalRotate(ct1, 1);
    });
    
    time_op("Rotate (by -1)", [&]() {
        result = cc->EvalRotate(ct1, -1);
    });
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Complex Operations" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Inner product / dot product
    time_op("EvalInnerProduct (depth 2)", [&]() {
        result = cc->EvalInnerProduct(ct1, ct2, (1 << 14));
    }, 5);
    
    // Polynomial evaluation
    std::vector<double> coeffs = {1.0, 0.5, 0.25, 0.125};
    time_op("EvalPoly (degree 3)", [&]() {
        result = cc->EvalPoly(ct1, coeffs);
    }, 5);
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmark Complete" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
