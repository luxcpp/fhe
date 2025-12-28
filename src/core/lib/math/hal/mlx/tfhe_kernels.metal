//==================================================================================
// Metal Kernels for GPU TFHE - Massively Parallel Operations
//==================================================================================

#include <metal_stdlib>
using namespace metal;

//==================================================================================
// Constants and Configuration
//==================================================================================

constant uint N [[function_constant(0)]];        // Ring dimension (e.g., 1024)
constant uint L [[function_constant(1)]];        // Decomposition digits
constant uint LOG_N [[function_constant(2)]];    // log2(N)
constant uint BASE_LOG [[function_constant(3)]]; // Decomposition base log
constant ulong Q [[function_constant(4)]];       // Ring modulus

//==================================================================================
// Modular Arithmetic (inline for performance)
//==================================================================================

inline ulong mulmod(ulong a, ulong b, ulong m) {
    // For Q < 2^32, we can use 64-bit multiplication
    return (a * b) % m;
}

inline ulong addmod(ulong a, ulong b, ulong m) {
    ulong sum = a + b;
    return (sum >= m) ? sum - m : sum;
}

inline ulong submod(ulong a, ulong b, ulong m) {
    return (a >= b) ? a - b : a + m - b;
}

//==================================================================================
// Batch NTT Kernel - Process multiple polynomials in parallel
//==================================================================================

// Each thread group processes one polynomial
// Threads within the group cooperate on butterfly operations

kernel void batchNTTForward(
    device long* polys [[buffer(0)]],           // [batch, N] polynomials
    constant long* twiddles [[buffer(1)]],       // [N] twiddle factors
    constant uint& batch [[buffer(2)]],          // Number of polynomials
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgSize [[threads_per_threadgroup]]
) {
    uint polyIdx = gid.y;  // Which polynomial
    uint elemIdx = gid.x;  // Which coefficient
    
    if (polyIdx >= batch || elemIdx >= N) return;
    
    device long* poly = polys + polyIdx * N;
    
    // Bit-reversal permutation (done once at start)
    uint rev = 0;
    uint temp = elemIdx;
    for (uint i = 0; i < LOG_N; ++i) {
        rev = (rev << 1) | (temp & 1);
        temp >>= 1;
    }
    
    if (elemIdx < rev) {
        long tmp = poly[elemIdx];
        poly[elemIdx] = poly[rev];
        poly[rev] = tmp;
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Cooley-Tukey butterfly stages
    for (uint len = 2; len <= N; len <<= 1) {
        uint halfLen = len >> 1;
        uint step = N / len;
        
        uint group = elemIdx / len;
        uint pos = elemIdx % len;
        
        if (pos < halfLen) {
            uint idx1 = group * len + pos;
            uint idx2 = idx1 + halfLen;
            
            ulong w = (ulong)twiddles[pos * step];
            ulong u = (ulong)poly[idx1] % Q;
            ulong v = mulmod((ulong)poly[idx2] % Q, w, Q);
            
            poly[idx1] = (long)addmod(u, v, Q);
            poly[idx2] = (long)submod(u, v, Q);
        }
        
        threadgroup_barrier(mem_flags::mem_device);
    }
}

kernel void batchNTTInverse(
    device long* polys [[buffer(0)]],
    constant long* invTwiddles [[buffer(1)]],
    constant ulong& nInv [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint polyIdx = gid.y;
    uint elemIdx = gid.x;
    
    if (polyIdx >= batch || elemIdx >= N) return;
    
    device long* poly = polys + polyIdx * N;
    
    // Same as forward but with inverse twiddles
    uint rev = 0;
    uint temp = elemIdx;
    for (uint i = 0; i < LOG_N; ++i) {
        rev = (rev << 1) | (temp & 1);
        temp >>= 1;
    }
    
    if (elemIdx < rev) {
        long tmp = poly[elemIdx];
        poly[elemIdx] = poly[rev];
        poly[rev] = tmp;
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    for (uint len = 2; len <= N; len <<= 1) {
        uint halfLen = len >> 1;
        uint step = N / len;
        
        uint group = elemIdx / len;
        uint pos = elemIdx % len;
        
        if (pos < halfLen) {
            uint idx1 = group * len + pos;
            uint idx2 = idx1 + halfLen;
            
            ulong w = (ulong)invTwiddles[pos * step];
            ulong u = (ulong)poly[idx1] % Q;
            ulong v = mulmod((ulong)poly[idx2] % Q, w, Q);
            
            poly[idx1] = (long)addmod(u, v, Q);
            poly[idx2] = (long)submod(u, v, Q);
        }
        
        threadgroup_barrier(mem_flags::mem_device);
    }
    
    // Scale by n^{-1}
    poly[elemIdx] = (long)mulmod((ulong)poly[elemIdx] % Q, nInv, Q);
}

//==================================================================================
// Batch External Product - Fused decompose + multiply + accumulate
//==================================================================================

// Process B external products in parallel
// Each thread handles one coefficient of one polynomial

kernel void batchExternalProduct(
    device const long* rlwe [[buffer(0)]],      // [B, 2, N] RLWE ciphertexts
    device const long* rgsw [[buffer(1)]],      // [B, 2, L, 2, N] RGSW ciphertexts
    device long* output [[buffer(2)]],          // [B, 2, N] output RLWE
    constant uint& batchSize [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint b = gid.z;      // Batch index
    uint comp = gid.y;   // Component (0 or 1)
    uint coeff = gid.x;  // Coefficient index
    
    if (b >= batchSize || comp >= 2 || coeff >= N) return;
    
    ulong acc = 0;
    ulong mask = (1UL << BASE_LOG) - 1;
    
    // For each RLWE component k (decomposition source)
    for (uint k = 0; k < 2; ++k) {
        // Get RLWE coefficient to decompose
        ulong rlweCoeff = (ulong)rlwe[b * 2 * N + k * N + coeff] % Q;
        
        // For each decomposition digit
        for (uint l = 0; l < L; ++l) {
            // Extract digit
            ulong digit = (rlweCoeff >> (l * BASE_LOG)) & mask;
            
            // Multiply by RGSW[k][l][comp][coeff] and accumulate
            // rgsw layout: [b, k, l, comp, coeff]
            ulong rgswVal = (ulong)rgsw[b * 2 * L * 2 * N + 
                                        k * L * 2 * N + 
                                        l * 2 * N + 
                                        comp * N + 
                                        coeff] % Q;
            
            acc = addmod(acc, mulmod(digit, rgswVal, Q), Q);
        }
    }
    
    output[b * 2 * N + comp * N + coeff] = (long)acc;
}

//==================================================================================
// Batch Blind Rotation - The core PBS operation
//==================================================================================

// This kernel handles the outer loop of blind rotation
// Each thread group processes one LWE ciphertext

kernel void batchBlindRotateInit(
    device const long* lwe [[buffer(0)]],       // [B, n+1] LWE ciphertexts
    device const long* testPoly [[buffer(1)]],  // [N] test polynomial
    device long* acc [[buffer(2)]],             // [B, 2, N] accumulators
    constant uint& batchSize [[buffer(3)]],
    constant uint& n [[buffer(4)]],             // LWE dimension
    uint3 gid [[thread_position_in_grid]]
) {
    uint b = gid.y;      // Batch index
    uint comp = gid.z;   // Component (0 or 1)
    uint coeff = gid.x;  // Coefficient
    
    if (b >= batchSize || comp >= 2 || coeff >= N) return;
    
    if (comp == 0) {
        // acc[0] = 0
        acc[b * 2 * N + coeff] = 0;
    } else {
        // acc[1] = X^{-b} * testPoly (negacyclic rotation)
        long bVal = lwe[b * (n + 1) + n];
        int shift = (int)(bVal % (long)(2 * N));
        if (shift < 0) shift += 2 * N;
        
        int srcIdx = ((int)coeff + shift) % (2 * (int)N);
        
        if (srcIdx < (int)N) {
            acc[b * 2 * N + N + coeff] = testPoly[srcIdx];
        } else {
            // Negacyclic: X^N = -1
            acc[b * 2 * N + N + coeff] = (long)((Q - (ulong)testPoly[srcIdx - N]) % Q);
        }
    }
}

// CMux operation: acc = acc + (X^a - 1) * RGSW(s) ⊗ acc
// This is called for each LWE mask coefficient

kernel void batchCMux(
    device long* acc [[buffer(0)]],             // [B, 2, N] accumulators
    device const long* bsk [[buffer(1)]],       // [B, n, 2, L, 2, N] bootstrap keys
    device const long* aVals [[buffer(2)]],     // [B] current a[i] values
    constant uint& batchSize [[buffer(3)]],
    constant uint& maskIdx [[buffer(4)]],       // Current mask index i
    uint3 gid [[thread_position_in_grid]]
) {
    uint b = gid.z;
    uint comp = gid.y;
    uint coeff = gid.x;
    
    if (b >= batchSize || comp >= 2 || coeff >= N) return;
    
    long aVal = aVals[b];
    if (aVal == 0) return;  // Skip if a[i] = 0
    
    // Get current accumulator value
    ulong accVal = (ulong)acc[b * 2 * N + comp * N + coeff] % Q;
    
    // Compute X^a * acc - acc (rotation by a, subtract original)
    int shift = (int)(aVal % (long)(2 * N));
    if (shift < 0) shift += 2 * N;
    
    int srcIdx = ((int)coeff - shift + 2 * (int)N) % (2 * (int)N);
    ulong rotated;
    
    if (srcIdx < (int)N) {
        rotated = (ulong)acc[b * 2 * N + comp * N + srcIdx] % Q;
    } else {
        rotated = (Q - (ulong)acc[b * 2 * N + comp * N + (srcIdx - N)]) % Q;
    }
    
    ulong diff = submod(rotated, accVal, Q);
    
    // External product with BK[i]
    ulong result = 0;
    ulong mask = (1UL << BASE_LOG) - 1;
    
    for (uint k = 0; k < 2; ++k) {
        ulong coeffVal = (k == comp) ? diff : 0;
        
        for (uint l = 0; l < L; ++l) {
            ulong digit = (coeffVal >> (l * BASE_LOG)) & mask;
            
            // bsk[b][i][k][l][comp][coeff]
            ulong bskVal = (ulong)bsk[b * 512 * 2 * L * 2 * N +  // Assuming n=512
                                      maskIdx * 2 * L * 2 * N +
                                      k * L * 2 * N +
                                      l * 2 * N +
                                      comp * N +
                                      coeff] % Q;
            
            result = addmod(result, mulmod(digit, bskVal, Q), Q);
        }
    }
    
    // acc += result
    acc[b * 2 * N + comp * N + coeff] = (long)addmod(accVal, result, Q);
}

//==================================================================================
// Batch Gate Execution - Apply gate operations to ciphertexts
//==================================================================================

// Prepare inputs for batch bootstrap
// Combines two LWE ciphertexts according to gate type

kernel void batchPrepareGateInputs(
    device const long* ct1 [[buffer(0)]],       // [B, n+1] first input
    device const long* ct2 [[buffer(1)]],       // [B, n+1] second input
    device long* combined [[buffer(2)]],         // [B, n+1] combined output
    constant uint& gateType [[buffer(3)]],       // Gate type enum
    constant uint& batchSize [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant ulong& mu [[buffer(6)]],            // Q/8
    uint3 gid [[thread_position_in_grid]]
) {
    uint b = gid.y;
    uint i = gid.x;
    
    if (b >= batchSize || i > n) return;
    
    long v1 = ct1[b * (n + 1) + i];
    long v2 = ct2[b * (n + 1) + i];
    long result;
    
    // Gate-specific input combination
    switch (gateType) {
        case 0: // AND: ct1 + ct2 + (-Q/8)
            if (i == n) {
                result = (long)addmod(addmod((ulong)v1 % Q, (ulong)v2 % Q, Q), 
                                      Q - mu, Q);
            } else {
                result = (long)addmod((ulong)v1 % Q, (ulong)v2 % Q, Q);
            }
            break;
            
        case 1: // OR: ct1 + ct2 + Q/8
            if (i == n) {
                result = (long)addmod(addmod((ulong)v1 % Q, (ulong)v2 % Q, Q),
                                      mu, Q);
            } else {
                result = (long)addmod((ulong)v1 % Q, (ulong)v2 % Q, Q);
            }
            break;
            
        case 2: // XOR: 2 * (ct1 + ct2)
            {
                ulong sum = addmod((ulong)v1 % Q, (ulong)v2 % Q, Q);
                result = (long)addmod(sum, sum, Q);  // 2 * sum
            }
            break;
            
        case 3: // NAND: -ct1 - ct2 + Q/8
            if (i == n) {
                result = (long)addmod(submod(Q, addmod((ulong)v1 % Q, (ulong)v2 % Q, Q), Q),
                                      mu, Q);
            } else {
                result = (long)submod(Q, addmod((ulong)v1 % Q, (ulong)v2 % Q, Q), Q);
            }
            break;
            
        case 4: // NOR: -ct1 - ct2 - Q/8
            if (i == n) {
                result = (long)submod(submod(Q, addmod((ulong)v1 % Q, (ulong)v2 % Q, Q), Q),
                                      mu, Q);
            } else {
                result = (long)submod(Q, addmod((ulong)v1 % Q, (ulong)v2 % Q, Q), Q);
            }
            break;
            
        case 5: // XNOR: -2 * (ct1 + ct2)
            {
                ulong sum = addmod((ulong)v1 % Q, (ulong)v2 % Q, Q);
                ulong doubled = addmod(sum, sum, Q);
                result = (long)submod(Q, doubled, Q);
            }
            break;
            
        default:
            result = v1;
    }
    
    combined[b * (n + 1) + i] = result;
}

//==================================================================================
// Key Switching Kernel
//==================================================================================

kernel void batchKeySwitch(
    device const long* rlwe [[buffer(0)]],      // [B, 2, N] RLWE input
    device const long* ksk [[buffer(1)]],       // [N, L_ks, n] key switching key
    device long* lwe [[buffer(2)]],             // [B, n+1] LWE output
    constant uint& batchSize [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& Lks [[buffer(5)]],           // KSK decomposition
    uint3 gid [[thread_position_in_grid]]
) {
    uint b = gid.y;
    uint i = gid.x;
    
    if (b >= batchSize || i > n) return;
    
    if (i < n) {
        // Compute LWE mask coefficient
        ulong acc = 0;
        
        for (uint j = 0; j < N; ++j) {
            ulong rlweCoeff = (ulong)rlwe[b * 2 * N + N + j] % Q;
            
            for (uint l = 0; l < Lks; ++l) {
                ulong digit = (rlweCoeff >> (l * BASE_LOG)) & ((1UL << BASE_LOG) - 1);
                ulong kskVal = (ulong)ksk[j * Lks * n + l * n + i] % Q;
                acc = addmod(acc, mulmod(digit, kskVal, Q), Q);
            }
        }
        
        lwe[b * (n + 1) + i] = (long)acc;
    } else {
        // Body: extract from RLWE
        lwe[b * (n + 1) + n] = rlwe[b * 2 * N];  // c0[0]
    }
}

//==================================================================================
// Utility Kernels
//==================================================================================

kernel void batchNegate(
    device long* cts [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    cts[gid] = (long)((Q - (ulong)cts[gid] % Q) % Q);
}

kernel void batchAdd(
    device const long* a [[buffer(0)]],
    device const long* b [[buffer(1)]],
    device long* c [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    c[gid] = (long)addmod((ulong)a[gid] % Q, (ulong)b[gid] % Q, Q);
}

kernel void batchSub(
    device const long* a [[buffer(0)]],
    device const long* b [[buffer(1)]],
    device long* c [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    c[gid] = (long)submod((ulong)a[gid] % Q, (ulong)b[gid] % Q, Q);
}
