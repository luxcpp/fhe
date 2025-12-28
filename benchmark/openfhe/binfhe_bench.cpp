// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// Benchmark suite for OpenFHE TFHE/BinFHE operations

#include <binfhe/binfhecontext.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>

using namespace lbcrypto;
using namespace std::chrono;

void run_benchmarks(BINFHE_PARAMSET paramSet, const std::string& paramName) {
    std::cout << "\n=== " << paramName << " ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    
    // Setup
    auto start = high_resolution_clock::now();
    auto cc = BinFHEContext();
    // Use method compatible with param set
    BINFHE_METHOD method = (paramSet == STD128_LMKCDEY) ? LMKCDEY : GINX;
    cc.GenerateBinFHEContext(paramSet, method);
    auto setupTime = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
    std::cout << "Context setup: " << setupTime << " ms" << std::endl;
    
    // Key generation
    start = high_resolution_clock::now();
    auto sk = cc.KeyGen();
    auto keygenTime = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
    std::cout << "Secret key gen: " << keygenTime << " ms" << std::endl;
    
    // Bootstrap key generation
    start = high_resolution_clock::now();
    cc.BTKeyGen(sk);
    auto btkeyTime = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
    std::cout << "Bootstrap key gen: " << btkeyTime << " ms" << std::endl;
    
    // Prepare ciphertexts (need independent ones for gate ops)
    std::vector<LWECiphertext> cts_a, cts_b;
    const int ITERS = 10;
    for (int i = 0; i < ITERS * 6; i++) {
        cts_a.push_back(cc.Encrypt(sk, 1));
        cts_b.push_back(cc.Encrypt(sk, 0));
    }
    
    // Encrypt benchmark
    start = high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        cc.Encrypt(sk, i % 2);
    }
    double encTime = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 100.0 / 1000.0;
    std::cout << "Encrypt: " << encTime << " ms" << std::endl;
    
    // Decrypt benchmark
    LWEPlaintext result;
    start = high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        cc.Decrypt(sk, cts_a[0], &result);
    }
    double decTime = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / 100.0 / 1000.0;
    std::cout << "Decrypt: " << decTime << " ms" << std::endl;
    
    // Gate operations (with bootstrapping)
    std::cout << "\n--- Gate Operations ---" << std::endl;
    int idx = 0;
    
    // AND
    start = high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        cc.EvalBinGate(AND, cts_a[idx], cts_b[idx]);
        idx++;
    }
    double andTime = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / (double)ITERS / 1000.0;
    std::cout << "AND: " << andTime << " ms" << std::endl;
    
    // OR
    start = high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        cc.EvalBinGate(OR, cts_a[idx], cts_b[idx]);
        idx++;
    }
    double orTime = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / (double)ITERS / 1000.0;
    std::cout << "OR: " << orTime << " ms" << std::endl;
    
    // XOR
    start = high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        cc.EvalBinGate(XOR, cts_a[idx], cts_b[idx]);
        idx++;
    }
    double xorTime = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / (double)ITERS / 1000.0;
    std::cout << "XOR: " << xorTime << " ms" << std::endl;
    
    // NAND
    start = high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        cc.EvalBinGate(NAND, cts_a[idx], cts_b[idx]);
        idx++;
    }
    double nandTime = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / (double)ITERS / 1000.0;
    std::cout << "NAND: " << nandTime << " ms" << std::endl;
    
    // NOT (unary, no independence requirement)
    start = high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        cc.EvalNOT(cts_a[i]);
    }
    double notTime = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / (double)ITERS / 1000.0;
    std::cout << "NOT: " << notTime << " ms" << std::endl;
    
    // NOR
    start = high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++) {
        cc.EvalBinGate(NOR, cts_a[idx], cts_b[idx]);
        idx++;
    }
    double norTime = duration_cast<microseconds>(high_resolution_clock::now() - start).count() / (double)ITERS / 1000.0;
    std::cout << "NOR: " << norTime << " ms" << std::endl;
}

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "OpenFHE BinFHE/TFHE Benchmark Suite" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // 128-bit security (NIST standard)
    run_benchmarks(STD128_LMKCDEY, "STD128_LMKCDEY (128-bit security)");
    run_benchmarks(STD128, "STD128 (128-bit security, GINX)");
    
    std::cout << "\n==========================================" << std::endl;
    std::cout << "Benchmark complete" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    return 0;
}
