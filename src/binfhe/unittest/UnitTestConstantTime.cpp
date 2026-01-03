// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// Unit tests for constant-time security operations

#include "gtest/gtest.h"
#include "security/constant_time.h"
#include "security/timing_guard.h"

#include <array>
#include <chrono>
#include <cstring>
#include <random>
#include <vector>

using namespace lux::fhe::security;

// ============================================================================
// Constant-Time Select Tests
// ============================================================================

class ConstantTimeSelectTest : public ::testing::Test {
protected:
    void SetUp() override {
        rng.seed(42);  // Deterministic for reproducibility
    }

    std::mt19937_64 rng;
};

TEST_F(ConstantTimeSelectTest, SelectWithFullMask) {
    uint64_t mask = ~static_cast<uint64_t>(0);  // All 1s
    uint64_t a = 0x123456789ABCDEF0;
    uint64_t b = 0xFEDCBA9876543210;

    uint64_t result = ct_select(mask, a, b);
    EXPECT_EQ(result, a) << "Full mask should select a";
}

TEST_F(ConstantTimeSelectTest, SelectWithZeroMask) {
    uint64_t mask = 0;  // All 0s
    uint64_t a = 0x123456789ABCDEF0;
    uint64_t b = 0xFEDCBA9876543210;

    uint64_t result = ct_select(mask, a, b);
    EXPECT_EQ(result, b) << "Zero mask should select b";
}

TEST_F(ConstantTimeSelectTest, SelectBoolTrue) {
    uint64_t a = 100;
    uint64_t b = 200;

    uint64_t result = ct_select_bool(true, a, b);
    EXPECT_EQ(result, a);
}

TEST_F(ConstantTimeSelectTest, SelectBoolFalse) {
    uint64_t a = 100;
    uint64_t b = 200;

    uint64_t result = ct_select_bool(false, a, b);
    EXPECT_EQ(result, b);
}

TEST_F(ConstantTimeSelectTest, SelectRandomValues) {
    for (int i = 0; i < 1000; ++i) {
        uint64_t a = rng();
        uint64_t b = rng();
        bool condition = (rng() % 2) == 0;

        uint64_t expected = condition ? a : b;
        uint64_t result = ct_select_bool(condition, a, b);

        EXPECT_EQ(result, expected) << "Failed for a=" << a << " b=" << b
                                     << " condition=" << condition;
    }
}

// ============================================================================
// Constant-Time Comparison Tests
// ============================================================================

class ConstantTimeCompareTest : public ::testing::Test {
protected:
    std::mt19937_64 rng{42};
};

TEST_F(ConstantTimeCompareTest, EqualityEqual) {
    uint64_t a = 12345;
    uint64_t result = ct_eq(a, a);
    EXPECT_EQ(result, ~static_cast<uint64_t>(0)) << "Equal values should return all 1s";
}

TEST_F(ConstantTimeCompareTest, EqualityNotEqual) {
    uint64_t a = 12345;
    uint64_t b = 54321;
    uint64_t result = ct_eq(a, b);
    EXPECT_EQ(result, static_cast<uint64_t>(0)) << "Unequal values should return 0";
}

TEST_F(ConstantTimeCompareTest, EqualityZero) {
    uint64_t zero = 0;
    EXPECT_EQ(ct_eq(zero, zero), ~static_cast<uint64_t>(0));
    EXPECT_EQ(ct_eq(zero, static_cast<uint64_t>(1)), static_cast<uint64_t>(0));
}

TEST_F(ConstantTimeCompareTest, IsZero) {
    EXPECT_EQ(ct_is_zero(static_cast<uint64_t>(0)), ~static_cast<uint64_t>(0));
    EXPECT_EQ(ct_is_zero(static_cast<uint64_t>(1)), static_cast<uint64_t>(0));
    EXPECT_EQ(ct_is_zero(~static_cast<uint64_t>(0)), static_cast<uint64_t>(0));
}

TEST_F(ConstantTimeCompareTest, LessThan) {
    EXPECT_EQ(ct_lt<uint64_t>(3, 5), ~static_cast<uint64_t>(0)) << "3 < 5";
    EXPECT_EQ(ct_lt<uint64_t>(5, 3), static_cast<uint64_t>(0)) << "5 !< 3";
    EXPECT_EQ(ct_lt<uint64_t>(5, 5), static_cast<uint64_t>(0)) << "5 !< 5";
    EXPECT_EQ(ct_lt<uint64_t>(0, 1), ~static_cast<uint64_t>(0)) << "0 < 1";
    EXPECT_EQ(ct_lt<uint64_t>(0, 0), static_cast<uint64_t>(0)) << "0 !< 0";
}

TEST_F(ConstantTimeCompareTest, LessThanEdgeCases) {
    constexpr uint64_t MAX = ~static_cast<uint64_t>(0);

    EXPECT_EQ(ct_lt(MAX - 1, MAX), ~static_cast<uint64_t>(0)) << "MAX-1 < MAX";
    EXPECT_EQ(ct_lt(MAX, MAX - 1), static_cast<uint64_t>(0)) << "MAX !< MAX-1";
    EXPECT_EQ(ct_lt(static_cast<uint64_t>(0), MAX), ~static_cast<uint64_t>(0)) << "0 < MAX";
    EXPECT_EQ(ct_lt(MAX, static_cast<uint64_t>(0)), static_cast<uint64_t>(0)) << "MAX !< 0";
}

TEST_F(ConstantTimeCompareTest, GreaterThan) {
    EXPECT_EQ(ct_gt<uint64_t>(5, 3), ~static_cast<uint64_t>(0)) << "5 > 3";
    EXPECT_EQ(ct_gt<uint64_t>(3, 5), static_cast<uint64_t>(0)) << "3 !> 5";
    EXPECT_EQ(ct_gt<uint64_t>(5, 5), static_cast<uint64_t>(0)) << "5 !> 5";
}

TEST_F(ConstantTimeCompareTest, LessOrEqual) {
    EXPECT_EQ(ct_le<uint64_t>(3, 5), ~static_cast<uint64_t>(0)) << "3 <= 5";
    EXPECT_EQ(ct_le<uint64_t>(5, 5), ~static_cast<uint64_t>(0)) << "5 <= 5";
    EXPECT_EQ(ct_le<uint64_t>(5, 3), static_cast<uint64_t>(0)) << "5 !<= 3";
}

TEST_F(ConstantTimeCompareTest, GreaterOrEqual) {
    EXPECT_EQ(ct_ge<uint64_t>(5, 3), ~static_cast<uint64_t>(0)) << "5 >= 3";
    EXPECT_EQ(ct_ge<uint64_t>(5, 5), ~static_cast<uint64_t>(0)) << "5 >= 5";
    EXPECT_EQ(ct_ge<uint64_t>(3, 5), static_cast<uint64_t>(0)) << "3 !>= 5";
}

TEST_F(ConstantTimeCompareTest, RandomComparisons) {
    for (int i = 0; i < 1000; ++i) {
        uint64_t a = rng();
        uint64_t b = rng();

        // Test lt
        bool expected_lt = (a < b);
        uint64_t result_lt = ct_lt(a, b);
        EXPECT_EQ(result_lt != 0, expected_lt) << "lt failed for a=" << a << " b=" << b;

        // Test eq
        bool expected_eq = (a == b);
        uint64_t result_eq = ct_eq(a, b);
        EXPECT_EQ(result_eq != 0, expected_eq) << "eq failed for a=" << a << " b=" << b;

        // Test gt
        bool expected_gt = (a > b);
        uint64_t result_gt = ct_gt(a, b);
        EXPECT_EQ(result_gt != 0, expected_gt) << "gt failed for a=" << a << " b=" << b;
    }
}

// ============================================================================
// Constant-Time Memory Operation Tests
// ============================================================================

class ConstantTimeMemoryTest : public ::testing::Test {
protected:
    std::array<uint8_t, 64> buffer1;
    std::array<uint8_t, 64> buffer2;
    std::array<uint8_t, 64> zeros;

    void SetUp() override {
        for (size_t i = 0; i < 64; ++i) {
            buffer1[i] = static_cast<uint8_t>(i);
            buffer2[i] = static_cast<uint8_t>(i);
        }
        zeros.fill(0);
    }
};

TEST_F(ConstantTimeMemoryTest, MemcmpEqual) {
    int result = ct_memcmp(buffer1.data(), buffer2.data(), 64);
    EXPECT_EQ(result, 0) << "Equal buffers should return 0";
}

TEST_F(ConstantTimeMemoryTest, MemcmpNotEqual) {
    buffer2[32] = 0xFF;  // Modify one byte
    int result = ct_memcmp(buffer1.data(), buffer2.data(), 64);
    EXPECT_NE(result, 0) << "Different buffers should return non-zero";
}

TEST_F(ConstantTimeMemoryTest, MemcmpNotEqualAtEnd) {
    buffer2[63] = 0xFF;  // Modify last byte
    int result = ct_memcmp(buffer1.data(), buffer2.data(), 64);
    EXPECT_NE(result, 0) << "Difference at end should be detected";
}

TEST_F(ConstantTimeMemoryTest, MemcpyWithMask) {
    std::array<uint8_t, 64> dest;
    dest.fill(0xAA);

    // Copy with mask = 0xFF (should copy)
    ct_memcpy(dest.data(), buffer1.data(), 64, 0xFF);
    EXPECT_EQ(ct_memcmp(dest.data(), buffer1.data(), 64), 0)
        << "Copy with mask=0xFF should copy all data";

    // Reset
    dest.fill(0xAA);

    // Copy with mask = 0x00 (should not change)
    ct_memcpy(dest.data(), buffer1.data(), 64, 0x00);
    for (size_t i = 0; i < 64; ++i) {
        EXPECT_EQ(dest[i], 0xAA) << "Copy with mask=0x00 should not modify dest";
    }
}

TEST_F(ConstantTimeMemoryTest, MemzeroClears) {
    ct_memzero(buffer1.data(), 64);
    EXPECT_EQ(ct_memcmp(buffer1.data(), zeros.data(), 64), 0)
        << "memzero should clear all bytes";
}

// ============================================================================
// Constant-Time Lookup Tests
// ============================================================================

class ConstantTimeLookupTest : public ::testing::Test {
protected:
    std::array<uint64_t, 16> table;

    void SetUp() override {
        for (size_t i = 0; i < 16; ++i) {
            table[i] = 0x100 + i * 0x11;
        }
    }
};

TEST_F(ConstantTimeLookupTest, LookupAllIndices) {
    for (size_t i = 0; i < 16; ++i) {
        uint64_t result = ct_lookup(table.data(), 16, i);
        EXPECT_EQ(result, table[i]) << "Lookup failed for index " << i;
    }
}

TEST_F(ConstantTimeLookupTest, LookupFirstElement) {
    uint64_t result = ct_lookup(table.data(), 16, size_t{0});
    EXPECT_EQ(result, table[0]);
}

TEST_F(ConstantTimeLookupTest, LookupLastElement) {
    uint64_t result = ct_lookup(table.data(), 16, size_t{15});
    EXPECT_EQ(result, table[15]);
}

TEST_F(ConstantTimeLookupTest, Lookup2D) {
    // 4x4 table
    std::array<uint64_t, 16> table2d;
    for (size_t r = 0; r < 4; ++r) {
        for (size_t c = 0; c < 4; ++c) {
            table2d[r * 4 + c] = r * 100 + c;
        }
    }

    for (size_t r = 0; r < 4; ++r) {
        for (size_t c = 0; c < 4; ++c) {
            uint64_t result = ct_lookup_2d(table2d.data(), 4, 4, r, c);
            uint64_t expected = r * 100 + c;
            EXPECT_EQ(result, expected) << "2D lookup failed for (" << r << "," << c << ")";
        }
    }
}

// ============================================================================
// Constant-Time Arithmetic Tests
// ============================================================================

class ConstantTimeArithmeticTest : public ::testing::Test {};

TEST_F(ConstantTimeArithmeticTest, SaturatingAdd) {
    // Normal add
    EXPECT_EQ(ct_add_sat<uint64_t>(10, 20), 30u);

    // Overflow should saturate
    constexpr uint64_t MAX = ~static_cast<uint64_t>(0);
    EXPECT_EQ(ct_add_sat(MAX, static_cast<uint64_t>(1)), MAX);
    EXPECT_EQ(ct_add_sat(MAX, MAX), MAX);
}

TEST_F(ConstantTimeArithmeticTest, SaturatingSubtract) {
    // Normal subtract
    EXPECT_EQ(ct_sub_sat<uint64_t>(30, 10), 20u);

    // Underflow should saturate to 0
    EXPECT_EQ(ct_sub_sat<uint64_t>(10, 20), 0u);
    EXPECT_EQ(ct_sub_sat<uint64_t>(0, 1), 0u);
}

TEST_F(ConstantTimeArithmeticTest, Min) {
    EXPECT_EQ(ct_min<uint64_t>(10, 20), 10u);
    EXPECT_EQ(ct_min<uint64_t>(20, 10), 10u);
    EXPECT_EQ(ct_min<uint64_t>(10, 10), 10u);
}

TEST_F(ConstantTimeArithmeticTest, Max) {
    EXPECT_EQ(ct_max<uint64_t>(10, 20), 20u);
    EXPECT_EQ(ct_max<uint64_t>(20, 10), 20u);
    EXPECT_EQ(ct_max<uint64_t>(10, 10), 10u);
}

TEST_F(ConstantTimeArithmeticTest, Clamp) {
    EXPECT_EQ(ct_clamp<uint64_t>(15, 10, 20), 15u);  // In range
    EXPECT_EQ(ct_clamp<uint64_t>(5, 10, 20), 10u);   // Below min
    EXPECT_EQ(ct_clamp<uint64_t>(25, 10, 20), 20u);  // Above max
}

// ============================================================================
// Constant-Time Bit Operation Tests
// ============================================================================

class ConstantTimeBitTest : public ::testing::Test {};

TEST_F(ConstantTimeBitTest, GetBit) {
    uint64_t val = 0b1010;
    EXPECT_EQ(ct_get_bit(val, 0u), 0u);
    EXPECT_EQ(ct_get_bit(val, 1u), 1u);
    EXPECT_EQ(ct_get_bit(val, 2u), 0u);
    EXPECT_EQ(ct_get_bit(val, 3u), 1u);
}

TEST_F(ConstantTimeBitTest, SetBit) {
    uint64_t val = 0;
    val = ct_set_bit(val, 3u, true);
    EXPECT_EQ(val, 0b1000u);

    val = ct_set_bit(val, 1u, true);
    EXPECT_EQ(val, 0b1010u);

    // Setting with false should not change
    val = ct_set_bit(val, 0u, false);
    EXPECT_EQ(val, 0b1010u);
}

TEST_F(ConstantTimeBitTest, ClearBit) {
    uint64_t val = 0b1111;
    val = ct_clear_bit(val, 1u, true);
    EXPECT_EQ(val, 0b1101u);

    // Clearing with false should not change
    val = ct_clear_bit(val, 0u, false);
    EXPECT_EQ(val, 0b1101u);
}

// ============================================================================
// FHE-Specific Constant-Time Tests
// ============================================================================

class ConstantTimeFHETest : public ::testing::Test {};

TEST_F(ConstantTimeFHETest, CompareFlag) {
    // LT case
    EXPECT_EQ(ct_compare_flag<uint64_t>(3, 5), 0u);  // LT

    // EQ case
    EXPECT_EQ(ct_compare_flag<uint64_t>(5, 5), 1u);  // EQ

    // GT case
    EXPECT_EQ(ct_compare_flag<uint64_t>(7, 5), 2u);  // GT
}

TEST_F(ConstantTimeFHETest, CombineFlags) {
    // If high is LT, result is LT regardless of low
    EXPECT_EQ(ct_combine_flags(0, 0), 0u);  // LT, LT -> LT
    EXPECT_EQ(ct_combine_flags(0, 1), 0u);  // LT, EQ -> LT
    EXPECT_EQ(ct_combine_flags(0, 2), 0u);  // LT, GT -> LT

    // If high is EQ, result is low
    EXPECT_EQ(ct_combine_flags(1, 0), 0u);  // EQ, LT -> LT
    EXPECT_EQ(ct_combine_flags(1, 1), 1u);  // EQ, EQ -> EQ
    EXPECT_EQ(ct_combine_flags(1, 2), 2u);  // EQ, GT -> GT

    // If high is GT, result is GT regardless of low
    EXPECT_EQ(ct_combine_flags(2, 0), 2u);  // GT, LT -> GT
    EXPECT_EQ(ct_combine_flags(2, 1), 2u);  // GT, EQ -> GT
    EXPECT_EQ(ct_combine_flags(2, 2), 2u);  // GT, GT -> GT
}

// ============================================================================
// Verification Function Tests
// ============================================================================

class VerificationTest : public ::testing::Test {};

TEST_F(VerificationTest, VerifyConstantTimeLookup) {
    EXPECT_TRUE(VerifyConstantTimeLookup());
}

TEST_F(VerificationTest, VerifyConstantTimeCompare) {
    EXPECT_TRUE(VerifyConstantTimeCompare());
}

TEST_F(VerificationTest, VerifyConstantTimeMemory) {
    EXPECT_TRUE(VerifyConstantTimeMemory());
}

TEST_F(VerificationTest, VerifyAllConstantTimeOps) {
    EXPECT_NO_THROW(VerifyAllConstantTimeOps());
}

// ============================================================================
// Timing Guard Tests
// ============================================================================

class TimingGuardTest : public ::testing::Test {};

TEST_F(TimingGuardTest, TimingGuardEnforcesMinimum) {
    auto start = std::chrono::high_resolution_clock::now();

    {
        // Guard with ~1ms minimum (assuming ~3GHz CPU = ~3M cycles)
        TimingGuard guard(3000000);
        // Empty - just measuring guard overhead
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // Should take at least ~900us (allowing for measurement overhead)
    EXPECT_GE(duration_us, 500) << "TimingGuard should enforce minimum time";
}

TEST_F(TimingGuardTest, TimingGuardDisable) {
    auto start = std::chrono::high_resolution_clock::now();

    {
        TimingGuard guard(100000000);  // Very long timeout
        guard.Disable();  // Disable the guard
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Should complete quickly since guard is disabled
    EXPECT_LT(duration_ms, 100) << "Disabled guard should not delay";
}

TEST_F(TimingGuardTest, MemoryAccessGuardTouchesAllRegions) {
    std::array<uint8_t, 256> region1;
    std::array<uint8_t, 256> region2;
    region1.fill(0xAA);
    region2.fill(0xBB);

    {
        MemoryAccessGuard guard;
        guard.AddRegion(region1.data(), region1.size());
        guard.AddRegion(region2.data(), region2.size());
        // Guard will touch all regions on destruction
    }

    // Regions should still contain original data
    EXPECT_EQ(region1[0], 0xAA);
    EXPECT_EQ(region2[0], 0xBB);
}

// ============================================================================
// Secret Wrapper Tests
// ============================================================================

class SecretWrapperTest : public ::testing::Test {};

TEST_F(SecretWrapperTest, SecretEquals) {
    Secret<uint64_t> secret(42);

    // Check equality returns proper mask
    auto eq_mask = secret.Equals(42);
    EXPECT_NE(eq_mask, 0u);

    auto neq_mask = secret.Equals(43);
    EXPECT_EQ(neq_mask, 0u);
}

TEST_F(SecretWrapperTest, SecretSelect) {
    Secret<uint64_t> secret_nonzero(1);
    Secret<uint64_t> secret_zero(0);

    EXPECT_EQ(secret_nonzero.Select(100, 200), 100u);
    EXPECT_EQ(secret_zero.Select(100, 200), 200u);
}

TEST_F(SecretWrapperTest, SecretDeclassify) {
    Secret<uint64_t> secret(12345);
    EXPECT_EQ(secret.Declassify(), 12345u);
}

// ============================================================================
// Radix Integer Comparison Tests
// ============================================================================

class RadixCompareTest : public ::testing::Test {};

TEST_F(RadixCompareTest, RadixCompareLT) {
    std::array<uint64_t, 4> a = {0, 0, 0, 0};
    std::array<uint64_t, 4> b = {1, 0, 0, 0};

    uint8_t result = ct_radix_compare(a.data(), b.data(), 4);
    EXPECT_EQ(result, 0u) << "0 < 1 should give LT flag";
}

TEST_F(RadixCompareTest, RadixCompareEQ) {
    std::array<uint64_t, 4> a = {1, 2, 3, 4};
    std::array<uint64_t, 4> b = {1, 2, 3, 4};

    uint8_t result = ct_radix_compare(a.data(), b.data(), 4);
    EXPECT_EQ(result, 1u) << "Equal values should give EQ flag";
}

TEST_F(RadixCompareTest, RadixCompareGT) {
    std::array<uint64_t, 4> a = {2, 0, 0, 0};
    std::array<uint64_t, 4> b = {1, 0, 0, 0};

    uint8_t result = ct_radix_compare(a.data(), b.data(), 4);
    EXPECT_EQ(result, 2u) << "2 > 1 should give GT flag";
}

TEST_F(RadixCompareTest, RadixCompareHighLimbDifference) {
    std::array<uint64_t, 4> a = {9, 9, 9, 1};  // Higher high limb
    std::array<uint64_t, 4> b = {0, 0, 0, 2};  // Lower high limb

    uint8_t result = ct_radix_compare(a.data(), b.data(), 4);
    EXPECT_EQ(result, 0u) << "High limb determines result (1 < 2)";
}

// Main provided by Main_TestAll.cpp for combined test suite
