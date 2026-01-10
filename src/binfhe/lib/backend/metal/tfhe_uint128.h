// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// 128-bit unsigned integer support for Metal TFHE kernels
// Self-contained - no external dependencies

#ifndef TFHE_UINT128_H
#define TFHE_UINT128_H

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// uint128 - 128-bit unsigned integer using two ulong limbs
// ============================================================================

struct uint128 {
    ulong lo;  // Low 64 bits
    ulong hi;  // High 64 bits

    // Default constructor
    uint128() : lo(0), hi(0) {}

    // Construct from single ulong
    uint128(ulong v) : lo(v), hi(0) {}

    // Construct from two ulongs
    uint128(ulong l, ulong h) : lo(l), hi(h) {}
};

// ============================================================================
// 64-bit x 64-bit -> 128-bit multiplication
// ============================================================================

METAL_FUNC uint128 mul64x64_fast(ulong a, ulong b) {
    // Split into 32-bit parts
    ulong a_lo = a & 0xFFFFFFFFUL;
    ulong a_hi = a >> 32;
    ulong b_lo = b & 0xFFFFFFFFUL;
    ulong b_hi = b >> 32;

    // Partial products
    ulong p0 = a_lo * b_lo;          // lo * lo
    ulong p1 = a_lo * b_hi;          // lo * hi
    ulong p2 = a_hi * b_lo;          // hi * lo
    ulong p3 = a_hi * b_hi;          // hi * hi

    // Accumulate with carry handling
    ulong mid = p1 + p2;
    ulong carry_mid = (mid < p1) ? 1UL : 0UL;  // Carry from mid overflow

    ulong lo = p0 + (mid << 32);
    ulong carry_lo = (lo < p0) ? 1UL : 0UL;

    ulong hi = p3 + (mid >> 32) + (carry_mid << 32) + carry_lo;

    return uint128(lo, hi);
}

// ============================================================================
// 128-bit mod 64-bit
// ============================================================================

METAL_FUNC ulong mod_128_64(uint128 x, ulong Q) {
    // Simple iterative reduction
    // For production, use Barrett or Montgomery reduction
    if (Q == 0) return 0;

    // If x fits in 64 bits, simple mod
    if (x.hi == 0) {
        return x.lo % Q;
    }

    // Reduce hi part first
    ulong r = x.hi % Q;

    // Now compute (r * 2^64 + x.lo) mod Q
    // Since 2^64 mod Q = (2^64 - Q * floor(2^64/Q))
    // We use: result = (r * (2^64 mod Q) + x.lo) mod Q

    // 2^64 mod Q - compute using the fact that 2^64 = Q * k + rem
    // For typical TFHE moduli, this can be simplified

    // Iterative approach for correctness
    for (int i = 0; i < 64; i++) {
        r = (r << 1);
        if (r >= Q) r -= Q;
    }

    // Add lo part
    ulong lo_mod = x.lo % Q;
    r = r + lo_mod;
    if (r >= Q) r -= Q;

    return r;
}

// ============================================================================
// Addition and comparison
// ============================================================================

METAL_FUNC uint128 add128(uint128 a, uint128 b) {
    ulong lo = a.lo + b.lo;
    ulong carry = (lo < a.lo) ? 1UL : 0UL;
    ulong hi = a.hi + b.hi + carry;
    return uint128(lo, hi);
}

METAL_FUNC bool lt128(uint128 a, uint128 b) {
    return (a.hi < b.hi) || (a.hi == b.hi && a.lo < b.lo);
}

METAL_FUNC bool gte128(uint128 a, uint128 b) {
    return !lt128(a, b);
}

// ============================================================================
// Operators for uint128
// ============================================================================

METAL_FUNC uint128 operator+(uint128 a, uint128 b) {
    return add128(a, b);
}

METAL_FUNC uint128 operator+(uint128 a, ulong b) {
    return add128(a, uint128(b));
}

METAL_FUNC uint128 operator+(ulong a, uint128 b) {
    return add128(uint128(a), b);
}

#endif // TFHE_UINT128_H
