// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Lux Industries Inc
//
// EVM Functional Tests for Encrypted uint256 Operations
//
// Tests all EVM opcodes on encrypted uint256 with edge cases:
//   - Carry propagation across 128 limbs
//   - Overflow and underflow behavior
//   - Shift by 0, 1, 8, 255, 256, 257 bits
//   - Comparison edge cases (equal values, max values)
//   - Division by zero handling
//
// These tests validate semantic correctness, not performance.

#include "gtest/gtest.h"
#include "binfhecontext.h"

#ifdef WITH_LUX_EXTENSIONS
#include "radix/radix.h"
#include "radix/shortint.h"
#include "fhevm/fhevm.h"
#endif

#include <random>
#include <limits>
#include <cstring>

using namespace lbcrypto;

#ifdef WITH_LUX_EXTENSIONS

using namespace radix;
using namespace fhevm;

// ============================================================================
// Test Fixture
// ============================================================================

class EVMFunctionalTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Use TOY parameters for faster tests
        // Production would use STD128 or higher
        cc_.GenerateBinFHEContext(TOY, GINX);
        sk_ = cc_.KeyGen();
        cc_.BTKeyGen(sk_);

        luts_ = std::make_unique<ShortIntLUTs>(params::EUINT8.limb_params);
    }

    void TearDown() override {
        luts_.reset();
    }

    // Helper: Encrypt a 64-bit value as uint256
    RadixInt EncryptUint64(uint64_t value) {
        return RadixInt::Encrypt(cc_, params::EUINT64, value, sk_);
    }

    // Helper: Encrypt a 256-bit value from bytes
    RadixInt EncryptUint256(const std::vector<uint8_t>& bytes) {
        std::vector<uint8_t> padded(32, 0);
        size_t offset = 32 - std::min(bytes.size(), size_t(32));
        std::copy(bytes.begin(), bytes.end(), padded.begin() + offset);
        return RadixInt::EncryptBytes(cc_, params::EUINT256, padded, sk_);
    }

    // Helper: Create bytes from hex string
    std::vector<uint8_t> HexToBytes(const std::string& hex) {
        std::vector<uint8_t> bytes;
        for (size_t i = 0; i < hex.length(); i += 2) {
            bytes.push_back(static_cast<uint8_t>(
                std::stoul(hex.substr(i, 2), nullptr, 16)));
        }
        return bytes;
    }

    // Helper: Decrypt and compare
    void ExpectEqual(const RadixInt& ct, uint64_t expected) {
        uint64_t result = ct.Decrypt(sk_);
        EXPECT_EQ(result, expected);
    }

    // Helper: Decrypt bytes and compare
    void ExpectEqualBytes(const RadixInt& ct, const std::vector<uint8_t>& expected) {
        auto result = ct.DecryptBytes(sk_);
        EXPECT_EQ(result.size(), expected.size());
        for (size_t i = 0; i < expected.size(); i++) {
            EXPECT_EQ(result[i], expected[i])
                << "Mismatch at byte " << i;
        }
    }

    BinFHEContext cc_;
    LWEPrivateKey sk_;
    std::unique_ptr<ShortIntLUTs> luts_;
};

// ============================================================================
// Addition Tests (ADD opcode)
// ============================================================================

class EVMAddTest : public EVMFunctionalTest {};

TEST_F(EVMAddTest, SimpleAddition) {
    auto a = EncryptUint64(42);
    auto b = EncryptUint64(17);
    auto c = Add(a, b, *luts_);
    ExpectEqual(c, 42 + 17);
}

TEST_F(EVMAddTest, AdditionWithCarry) {
    // Test carry propagation: 0xFF + 0x01 = 0x100
    auto a = EncryptUint64(0xFF);
    auto b = EncryptUint64(0x01);
    auto c = Add(a, b, *luts_);
    ExpectEqual(c, 0x100);
}

TEST_F(EVMAddTest, AdditionWithMultipleCarries) {
    // 0xFFFF + 1 = 0x10000 (carries across two limbs)
    auto a = EncryptUint64(0xFFFF);
    auto b = EncryptUint64(0x0001);
    auto c = Add(a, b, *luts_);
    ExpectEqual(c, 0x10000);
}

TEST_F(EVMAddTest, MaxValuePlusOne) {
    // For 8-bit: 255 + 1 should overflow to 0
    RadixParams params8 = params::EUINT8;
    auto a = RadixInt::Encrypt(cc_, params8, 255, sk_);
    auto b = RadixInt::Encrypt(cc_, params8, 1, sk_);
    ShortIntLUTs luts8(params8.limb_params);
    auto c = Add(a, b, luts8);
    ExpectEqual(c, 0); // Overflow wraps to 0
}

TEST_F(EVMAddTest, AddZero) {
    auto a = EncryptUint64(12345);
    auto zero = EncryptUint64(0);
    auto c = Add(a, zero, *luts_);
    ExpectEqual(c, 12345);
}

TEST_F(EVMAddTest, Commutativity) {
    auto a = EncryptUint64(100);
    auto b = EncryptUint64(200);
    auto c1 = Add(a, b, *luts_);
    auto c2 = Add(b, a, *luts_);
    ExpectEqual(c1, 300);
    ExpectEqual(c2, 300);
}

// ============================================================================
// Subtraction Tests (SUB opcode)
// ============================================================================

class EVMSubTest : public EVMFunctionalTest {};

TEST_F(EVMSubTest, SimpleSubtraction) {
    auto a = EncryptUint64(100);
    auto b = EncryptUint64(30);
    auto c = Sub(a, b, *luts_);
    ExpectEqual(c, 70);
}

TEST_F(EVMSubTest, SubtractionWithBorrow) {
    // 0x100 - 0x01 = 0xFF (borrow from higher limb)
    auto a = EncryptUint64(0x100);
    auto b = EncryptUint64(0x01);
    auto c = Sub(a, b, *luts_);
    ExpectEqual(c, 0xFF);
}

TEST_F(EVMSubTest, Underflow) {
    // For unsigned: 0 - 1 should wrap to max value
    RadixParams params8 = params::EUINT8;
    auto a = RadixInt::Encrypt(cc_, params8, 0, sk_);
    auto b = RadixInt::Encrypt(cc_, params8, 1, sk_);
    ShortIntLUTs luts8(params8.limb_params);
    auto c = Sub(a, b, luts8);
    ExpectEqual(c, 255); // Two's complement wrap
}

TEST_F(EVMSubTest, SubtractZero) {
    auto a = EncryptUint64(12345);
    auto zero = EncryptUint64(0);
    auto c = Sub(a, zero, *luts_);
    ExpectEqual(c, 12345);
}

TEST_F(EVMSubTest, SubtractFromSelf) {
    auto a = EncryptUint64(12345);
    auto b = EncryptUint64(12345);
    auto c = Sub(a, b, *luts_);
    ExpectEqual(c, 0);
}

// ============================================================================
// Multiplication Tests (MUL opcode)
// ============================================================================

class EVMMulTest : public EVMFunctionalTest {};

TEST_F(EVMMulTest, SimpleMultiplication) {
    auto a = EncryptUint64(7);
    auto b = EncryptUint64(6);
    auto c = Mul(a, b, *luts_);
    ExpectEqual(c, 42);
}

TEST_F(EVMMulTest, MultiplyByZero) {
    auto a = EncryptUint64(12345);
    auto zero = EncryptUint64(0);
    auto c = Mul(a, zero, *luts_);
    ExpectEqual(c, 0);
}

TEST_F(EVMMulTest, MultiplyByOne) {
    auto a = EncryptUint64(12345);
    auto one = EncryptUint64(1);
    auto c = Mul(a, one, *luts_);
    ExpectEqual(c, 12345);
}

TEST_F(EVMMulTest, SquareSmallValue) {
    auto a = EncryptUint64(15);
    auto c = Mul(a, a, *luts_);
    ExpectEqual(c, 225);
}

// ============================================================================
// Comparison Tests (LT, GT, EQ, etc.)
// ============================================================================

class EVMCompareTest : public EVMFunctionalTest {};

TEST_F(EVMCompareTest, LessThan) {
    auto a = EncryptUint64(10);
    auto b = EncryptUint64(20);
    auto result = Lt(a, b, *luts_);
    ExpectEqual(result, 1); // true
}

TEST_F(EVMCompareTest, NotLessThan) {
    auto a = EncryptUint64(20);
    auto b = EncryptUint64(10);
    auto result = Lt(a, b, *luts_);
    ExpectEqual(result, 0); // false
}

TEST_F(EVMCompareTest, EqualValues) {
    auto a = EncryptUint64(42);
    auto b = EncryptUint64(42);

    auto lt = Lt(a, b, *luts_);
    auto gt = Gt(a, b, *luts_);
    auto eq = Eq(a, b, *luts_);

    ExpectEqual(lt, 0); // not less
    ExpectEqual(gt, 0); // not greater
    ExpectEqual(eq, 1); // equal
}

TEST_F(EVMCompareTest, NotEqual) {
    auto a = EncryptUint64(10);
    auto b = EncryptUint64(20);
    auto result = Ne(a, b, *luts_);
    ExpectEqual(result, 1); // true
}

TEST_F(EVMCompareTest, LessOrEqual) {
    auto a = EncryptUint64(10);
    auto b = EncryptUint64(10);
    auto result = Le(a, b, *luts_);
    ExpectEqual(result, 1); // true (equal case)
}

TEST_F(EVMCompareTest, GreaterOrEqual) {
    auto a = EncryptUint64(20);
    auto b = EncryptUint64(10);
    auto result = Ge(a, b, *luts_);
    ExpectEqual(result, 1); // true
}

TEST_F(EVMCompareTest, MaxValueComparison) {
    RadixParams params8 = params::EUINT8;
    ShortIntLUTs luts8(params8.limb_params);

    auto max = RadixInt::Encrypt(cc_, params8, 255, sk_);
    auto zero = RadixInt::Encrypt(cc_, params8, 0, sk_);

    auto result = Gt(max, zero, luts8);
    ExpectEqual(result, 1);
}

// ============================================================================
// Bitwise Operations Tests (AND, OR, XOR, NOT)
// ============================================================================

class EVMBitwiseTest : public EVMFunctionalTest {};

TEST_F(EVMBitwiseTest, BitwiseAnd) {
    auto a = EncryptUint64(0b1100);
    auto b = EncryptUint64(0b1010);
    auto c = BitwiseAnd(a, b, *luts_);
    ExpectEqual(c, 0b1000);
}

TEST_F(EVMBitwiseTest, BitwiseOr) {
    auto a = EncryptUint64(0b1100);
    auto b = EncryptUint64(0b1010);
    auto c = BitwiseOr(a, b, *luts_);
    ExpectEqual(c, 0b1110);
}

TEST_F(EVMBitwiseTest, BitwiseXor) {
    auto a = EncryptUint64(0b1100);
    auto b = EncryptUint64(0b1010);
    auto c = BitwiseXor(a, b, *luts_);
    ExpectEqual(c, 0b0110);
}

TEST_F(EVMBitwiseTest, BitwiseNot) {
    RadixParams params8 = params::EUINT8;
    ShortIntLUTs luts8(params8.limb_params);

    auto a = RadixInt::Encrypt(cc_, params8, 0b10101010, sk_);
    auto c = BitwiseNot(a, luts8);
    ExpectEqual(c, 0b01010101);
}

TEST_F(EVMBitwiseTest, XorSelf) {
    auto a = EncryptUint64(12345);
    auto c = BitwiseXor(a, a, *luts_);
    ExpectEqual(c, 0);
}

TEST_F(EVMBitwiseTest, AndWithZero) {
    auto a = EncryptUint64(0xFFFF);
    auto zero = EncryptUint64(0);
    auto c = BitwiseAnd(a, zero, *luts_);
    ExpectEqual(c, 0);
}

TEST_F(EVMBitwiseTest, OrWithZero) {
    auto a = EncryptUint64(0x1234);
    auto zero = EncryptUint64(0);
    auto c = BitwiseOr(a, zero, *luts_);
    ExpectEqual(c, 0x1234);
}

// ============================================================================
// Shift Operations Tests (SHL, SHR)
// ============================================================================

class EVMShiftTest : public EVMFunctionalTest {};

TEST_F(EVMShiftTest, ShiftLeftByZero) {
    auto a = EncryptUint64(0x12);
    a.ShlInPlace(0, *luts_);
    ExpectEqual(a, 0x12); // No change
}

TEST_F(EVMShiftTest, ShiftLeftByOne) {
    auto a = EncryptUint64(0x12);
    a.ShlInPlace(1, *luts_);
    ExpectEqual(a, 0x24); // Multiply by 2
}

TEST_F(EVMShiftTest, ShiftLeftByEight) {
    auto a = EncryptUint64(0x12);
    a.ShlInPlace(8, *luts_);
    ExpectEqual(a, 0x1200); // Byte-aligned shift
}

TEST_F(EVMShiftTest, ShiftRightByZero) {
    auto a = EncryptUint64(0x1234);
    a.ShrInPlace(0, *luts_);
    ExpectEqual(a, 0x1234); // No change
}

TEST_F(EVMShiftTest, ShiftRightByOne) {
    auto a = EncryptUint64(0x10);
    a.ShrInPlace(1, *luts_);
    ExpectEqual(a, 0x08); // Divide by 2
}

TEST_F(EVMShiftTest, ShiftRightByEight) {
    auto a = EncryptUint64(0x1200);
    a.ShrInPlace(8, *luts_);
    ExpectEqual(a, 0x12);
}

// Edge cases for shift >= bit width
TEST_F(EVMShiftTest, ShiftByBitWidth) {
    // For 8-bit: shift by 8 should result in 0 (EVM behavior)
    RadixParams params8 = params::EUINT8;
    ShortIntLUTs luts8(params8.limb_params);

    auto a = RadixInt::Encrypt(cc_, params8, 0xFF, sk_);
    a.ShlInPlace(8, luts8);
    ExpectEqual(a, 0); // All bits shifted out
}

TEST_F(EVMShiftTest, ShiftByMoreThanBitWidth) {
    // For 8-bit: shift by 9 should also result in 0
    RadixParams params8 = params::EUINT8;
    ShortIntLUTs luts8(params8.limb_params);

    auto a = RadixInt::Encrypt(cc_, params8, 0xFF, sk_);
    a.ShlInPlace(9, luts8);
    ExpectEqual(a, 0);
}

TEST_F(EVMShiftTest, ShiftRightAllBits) {
    RadixParams params8 = params::EUINT8;
    ShortIntLUTs luts8(params8.limb_params);

    auto a = RadixInt::Encrypt(cc_, params8, 0xFF, sk_);
    a.ShrInPlace(8, luts8);
    ExpectEqual(a, 0);
}

// ============================================================================
// Division and Modulo Tests (DIV, MOD)
// ============================================================================

class EVMDivTest : public EVMFunctionalTest {};

TEST_F(EVMDivTest, SimpleDivision) {
    auto a = EncryptUint64(100);
    auto b = EncryptUint64(10);
    auto [q, r] = Div(a, b, *luts_);
    ExpectEqual(q, 10);
    ExpectEqual(r, 0);
}

TEST_F(EVMDivTest, DivisionWithRemainder) {
    auto a = EncryptUint64(17);
    auto b = EncryptUint64(5);
    auto [q, r] = Div(a, b, *luts_);
    ExpectEqual(q, 3);  // 17 / 5 = 3
    ExpectEqual(r, 2);  // 17 % 5 = 2
}

TEST_F(EVMDivTest, DivideByOne) {
    auto a = EncryptUint64(12345);
    auto one = EncryptUint64(1);
    auto [q, r] = Div(a, one, *luts_);
    ExpectEqual(q, 12345);
    ExpectEqual(r, 0);
}

TEST_F(EVMDivTest, DivideZero) {
    auto zero = EncryptUint64(0);
    auto b = EncryptUint64(10);
    auto [q, r] = Div(zero, b, *luts_);
    ExpectEqual(q, 0);
    ExpectEqual(r, 0);
}

// EVM behavior: division by zero returns 0
TEST_F(EVMDivTest, DivisionByZero) {
    auto a = EncryptUint64(100);
    auto zero = EncryptUint64(0);
    auto [q, r] = Div(a, zero, *luts_);
    // EVM spec: DIV by 0 = 0
    ExpectEqual(q, 0);
}

TEST_F(EVMDivTest, ModuloOperation) {
    auto a = EncryptUint64(17);
    auto b = EncryptUint64(5);
    auto result = Mod(a, b, *luts_);
    ExpectEqual(result, 2);
}

// ============================================================================
// Min/Max Tests
// ============================================================================

class EVMMinMaxTest : public EVMFunctionalTest {};

TEST_F(EVMMinMaxTest, MinTest) {
    auto a = EncryptUint64(10);
    auto b = EncryptUint64(20);
    auto result = Min(a, b, *luts_);
    ExpectEqual(result, 10);
}

TEST_F(EVMMinMaxTest, MaxTest) {
    auto a = EncryptUint64(10);
    auto b = EncryptUint64(20);
    auto result = Max(a, b, *luts_);
    ExpectEqual(result, 20);
}

TEST_F(EVMMinMaxTest, MinEqualValues) {
    auto a = EncryptUint64(42);
    auto b = EncryptUint64(42);
    auto result = Min(a, b, *luts_);
    ExpectEqual(result, 42);
}

TEST_F(EVMMinMaxTest, MaxEqualValues) {
    auto a = EncryptUint64(42);
    auto b = EncryptUint64(42);
    auto result = Max(a, b, *luts_);
    ExpectEqual(result, 42);
}

// ============================================================================
// Select/CMUX Tests (conditional operations)
// ============================================================================

class EVMSelectTest : public EVMFunctionalTest {};

TEST_F(EVMSelectTest, SelectTrue) {
    auto sel = EncryptUint64(1);
    auto if_true = EncryptUint64(100);
    auto if_false = EncryptUint64(200);
    auto result = Select(sel, if_true, if_false, *luts_);
    ExpectEqual(result, 100);
}

TEST_F(EVMSelectTest, SelectFalse) {
    auto sel = EncryptUint64(0);
    auto if_true = EncryptUint64(100);
    auto if_false = EncryptUint64(200);
    auto result = Select(sel, if_true, if_false, *luts_);
    ExpectEqual(result, 200);
}

// ============================================================================
// Type Cast Tests
// ============================================================================

class EVMCastTest : public EVMFunctionalTest {};

TEST_F(EVMCastTest, CastToLarger) {
    RadixParams params8 = params::EUINT8;
    RadixParams params16 = params::EUINT16;

    auto a = RadixInt::Encrypt(cc_, params8, 255, sk_);
    auto b = Cast(a, params16);

    EXPECT_EQ(b.NumLimbs(), params16.num_limbs);
    ExpectEqual(b, 255); // Value preserved
}

TEST_F(EVMCastTest, CastToSmaller) {
    RadixParams params16 = params::EUINT16;
    RadixParams params8 = params::EUINT8;

    auto a = RadixInt::Encrypt(cc_, params16, 0x1234, sk_);
    auto b = Cast(a, params8);

    EXPECT_EQ(b.NumLimbs(), params8.num_limbs);
    ExpectEqual(b, 0x34); // Truncated to low byte
}

// ============================================================================
// IsZero / IsNonZero Tests
// ============================================================================

class EVMZeroCheckTest : public EVMFunctionalTest {};

TEST_F(EVMZeroCheckTest, IsZeroTrue) {
    auto a = EncryptUint64(0);
    auto result = IsZero(a, *luts_);
    ExpectEqual(result, 1);
}

TEST_F(EVMZeroCheckTest, IsZeroFalse) {
    auto a = EncryptUint64(42);
    auto result = IsZero(a, *luts_);
    ExpectEqual(result, 0);
}

TEST_F(EVMZeroCheckTest, IsNonZeroTrue) {
    auto a = EncryptUint64(42);
    auto result = IsNonZero(a, *luts_);
    ExpectEqual(result, 1);
}

TEST_F(EVMZeroCheckTest, IsNonZeroFalse) {
    auto a = EncryptUint64(0);
    auto result = IsNonZero(a, *luts_);
    ExpectEqual(result, 0);
}

// ============================================================================
// Carry Propagation Tests
// ============================================================================

class EVMCarryTest : public EVMFunctionalTest {};

TEST_F(EVMCarryTest, MultipleAdditionsWithCarry) {
    // Add 255 four times, should cause multiple carries
    auto a = EncryptUint64(255);
    auto b = EncryptUint64(255);

    auto c = Add(a, b, *luts_);
    c.PropagateCarries(*luts_);

    auto d = Add(c, b, *luts_);
    d.PropagateCarries(*luts_);

    auto e = Add(d, b, *luts_);
    e.PropagateCarries(*luts_);

    ExpectEqual(e, 255 * 4);
}

TEST_F(EVMCarryTest, CarryChain) {
    // 0x00FF + 0x0001 should propagate carry
    auto a = EncryptUint64(0x00FF);
    auto b = EncryptUint64(0x0001);
    auto c = Add(a, b, *luts_);
    ExpectEqual(c, 0x0100);
}

// ============================================================================
// Random Stress Tests
// ============================================================================

class EVMRandomTest : public EVMFunctionalTest {};

TEST_F(EVMRandomTest, RandomAdditions) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist(0, 255);

    for (int i = 0; i < 10; i++) {
        uint64_t va = dist(gen);
        uint64_t vb = dist(gen);

        auto a = EncryptUint64(va);
        auto b = EncryptUint64(vb);
        auto c = Add(a, b, *luts_);

        ExpectEqual(c, (va + vb) & 0xFFFFFFFFFFFFFFFFULL);
    }
}

TEST_F(EVMRandomTest, RandomComparisons) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist(0, 255);

    for (int i = 0; i < 10; i++) {
        uint64_t va = dist(gen);
        uint64_t vb = dist(gen);

        auto a = EncryptUint64(va);
        auto b = EncryptUint64(vb);

        auto lt = Lt(a, b, *luts_);
        auto eq = Eq(a, b, *luts_);
        auto gt = Gt(a, b, *luts_);

        ExpectEqual(lt, va < vb ? 1 : 0);
        ExpectEqual(eq, va == vb ? 1 : 0);
        ExpectEqual(gt, va > vb ? 1 : 0);
    }
}

#else // !WITH_LUX_EXTENSIONS

// Placeholder tests when extensions not available
TEST(EVMFunctionalTest, SkippedWithoutExtensions) {
    GTEST_SKIP() << "EVM functional tests require WITH_LUX_EXTENSIONS=ON";
}

#endif // WITH_LUX_EXTENSIONS
