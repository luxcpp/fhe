// SPDX-License-Identifier: BSD-3-Clause
pragma solidity ^0.8.20;

import {FheOps, Precompile} from "./Precompile.sol";

/// @title Encrypted Types
/// @notice Handles representing encrypted values stored off-chain
type ebool is uint256;
type euint8 is uint256;
type euint16 is uint256;
type euint32 is uint256;
type euint64 is uint256;
type euint128 is uint256;
type euint256 is uint256;
type eaddress is uint256;

/// @title Input Types for Encryption
/// @notice User-provided encrypted inputs with security zone
struct inEbool {
    bytes data;
    int32 securityZone;
}

struct inEuint8 {
    bytes data;
    int32 securityZone;
}

struct inEuint16 {
    bytes data;
    int32 securityZone;
}

struct inEuint32 {
    bytes data;
    int32 securityZone;
}

struct inEuint64 {
    bytes data;
    int32 securityZone;
}

struct inEuint128 {
    bytes data;
    int32 securityZone;
}

struct inEuint256 {
    bytes data;
    int32 securityZone;
}

struct inEaddress {
    bytes data;
    int32 securityZone;
}

/// @title Type Constants
library Types {
    uint8 internal constant EUINT8 = 0;
    uint8 internal constant EUINT16 = 1;
    uint8 internal constant EUINT32 = 2;
    uint8 internal constant EUINT64 = 3;
    uint8 internal constant EUINT128 = 4;
    uint8 internal constant EUINT256 = 5;
    uint8 internal constant EADDRESS = 12;
    uint8 internal constant EBOOL = 13;
}

/// @title FHE Library
/// @notice Main library for FHE operations in Solidity
/// @dev All operations call the FHE precompile at address 0x80
library FHE {
    
    // ============ Initialization Checks ============
    
    function isInitialized(ebool v) internal pure returns (bool) {
        return ebool.unwrap(v) != 0;
    }
    
    function isInitialized(euint8 v) internal pure returns (bool) {
        return euint8.unwrap(v) != 0;
    }
    
    function isInitialized(euint16 v) internal pure returns (bool) {
        return euint16.unwrap(v) != 0;
    }
    
    function isInitialized(euint32 v) internal pure returns (bool) {
        return euint32.unwrap(v) != 0;
    }
    
    function isInitialized(euint64 v) internal pure returns (bool) {
        return euint64.unwrap(v) != 0;
    }
    
    function isInitialized(euint128 v) internal pure returns (bool) {
        return euint128.unwrap(v) != 0;
    }
    
    function isInitialized(euint256 v) internal pure returns (bool) {
        return euint256.unwrap(v) != 0;
    }

    // ============ Encryption (from plaintext) ============
    
    function asEbool(bool value) internal returns (ebool) {
        return ebool.wrap(_trivialEncrypt(value ? 1 : 0, Types.EBOOL, 0));
    }
    
    function asEuint8(uint8 value) internal returns (euint8) {
        return euint8.wrap(_trivialEncrypt(value, Types.EUINT8, 0));
    }
    
    function asEuint16(uint16 value) internal returns (euint16) {
        return euint16.wrap(_trivialEncrypt(value, Types.EUINT16, 0));
    }
    
    function asEuint32(uint32 value) internal returns (euint32) {
        return euint32.wrap(_trivialEncrypt(value, Types.EUINT32, 0));
    }
    
    function asEuint64(uint64 value) internal returns (euint64) {
        return euint64.wrap(_trivialEncrypt(value, Types.EUINT64, 0));
    }
    
    function asEuint128(uint128 value) internal returns (euint128) {
        return euint128.wrap(_trivialEncrypt(value, Types.EUINT128, 0));
    }
    
    function asEuint256(uint256 value) internal returns (euint256) {
        return euint256.wrap(_trivialEncrypt(value, Types.EUINT256, 0));
    }

    // ============ Input Verification ============
    
    function asEbool(inEbool memory input) internal returns (ebool) {
        return ebool.wrap(_verify(input.data, Types.EBOOL, input.securityZone));
    }
    
    function asEuint8(inEuint8 memory input) internal returns (euint8) {
        return euint8.wrap(_verify(input.data, Types.EUINT8, input.securityZone));
    }
    
    function asEuint16(inEuint16 memory input) internal returns (euint16) {
        return euint16.wrap(_verify(input.data, Types.EUINT16, input.securityZone));
    }
    
    function asEuint32(inEuint32 memory input) internal returns (euint32) {
        return euint32.wrap(_verify(input.data, Types.EUINT32, input.securityZone));
    }
    
    function asEuint64(inEuint64 memory input) internal returns (euint64) {
        return euint64.wrap(_verify(input.data, Types.EUINT64, input.securityZone));
    }
    
    function asEuint128(inEuint128 memory input) internal returns (euint128) {
        return euint128.wrap(_verify(input.data, Types.EUINT128, input.securityZone));
    }
    
    function asEuint256(inEuint256 memory input) internal returns (euint256) {
        return euint256.wrap(_verify(input.data, Types.EUINT256, input.securityZone));
    }

    // ============ Arithmetic Operations ============
    
    function add(euint8 a, euint8 b) internal returns (euint8) {
        return euint8.wrap(_add(Types.EUINT8, euint8.unwrap(a), euint8.unwrap(b)));
    }
    
    function add(euint16 a, euint16 b) internal returns (euint16) {
        return euint16.wrap(_add(Types.EUINT16, euint16.unwrap(a), euint16.unwrap(b)));
    }
    
    function add(euint32 a, euint32 b) internal returns (euint32) {
        return euint32.wrap(_add(Types.EUINT32, euint32.unwrap(a), euint32.unwrap(b)));
    }
    
    function add(euint64 a, euint64 b) internal returns (euint64) {
        return euint64.wrap(_add(Types.EUINT64, euint64.unwrap(a), euint64.unwrap(b)));
    }
    
    function sub(euint8 a, euint8 b) internal returns (euint8) {
        return euint8.wrap(_sub(Types.EUINT8, euint8.unwrap(a), euint8.unwrap(b)));
    }
    
    function sub(euint16 a, euint16 b) internal returns (euint16) {
        return euint16.wrap(_sub(Types.EUINT16, euint16.unwrap(a), euint16.unwrap(b)));
    }
    
    function sub(euint32 a, euint32 b) internal returns (euint32) {
        return euint32.wrap(_sub(Types.EUINT32, euint32.unwrap(a), euint32.unwrap(b)));
    }
    
    function sub(euint64 a, euint64 b) internal returns (euint64) {
        return euint64.wrap(_sub(Types.EUINT64, euint64.unwrap(a), euint64.unwrap(b)));
    }
    
    function mul(euint8 a, euint8 b) internal returns (euint8) {
        return euint8.wrap(_mul(Types.EUINT8, euint8.unwrap(a), euint8.unwrap(b)));
    }
    
    function mul(euint16 a, euint16 b) internal returns (euint16) {
        return euint16.wrap(_mul(Types.EUINT16, euint16.unwrap(a), euint16.unwrap(b)));
    }
    
    function mul(euint32 a, euint32 b) internal returns (euint32) {
        return euint32.wrap(_mul(Types.EUINT32, euint32.unwrap(a), euint32.unwrap(b)));
    }
    
    function mul(euint64 a, euint64 b) internal returns (euint64) {
        return euint64.wrap(_mul(Types.EUINT64, euint64.unwrap(a), euint64.unwrap(b)));
    }

    // ============ Comparison Operations ============
    
    function lt(euint8 a, euint8 b) internal returns (ebool) {
        return ebool.wrap(_lt(Types.EUINT8, euint8.unwrap(a), euint8.unwrap(b)));
    }
    
    function lt(euint32 a, euint32 b) internal returns (ebool) {
        return ebool.wrap(_lt(Types.EUINT32, euint32.unwrap(a), euint32.unwrap(b)));
    }
    
    function lt(euint64 a, euint64 b) internal returns (ebool) {
        return ebool.wrap(_lt(Types.EUINT64, euint64.unwrap(a), euint64.unwrap(b)));
    }
    
    function gt(euint8 a, euint8 b) internal returns (ebool) {
        return ebool.wrap(_gt(Types.EUINT8, euint8.unwrap(a), euint8.unwrap(b)));
    }
    
    function gt(euint32 a, euint32 b) internal returns (ebool) {
        return ebool.wrap(_gt(Types.EUINT32, euint32.unwrap(a), euint32.unwrap(b)));
    }
    
    function gt(euint64 a, euint64 b) internal returns (ebool) {
        return ebool.wrap(_gt(Types.EUINT64, euint64.unwrap(a), euint64.unwrap(b)));
    }
    
    function eq(euint8 a, euint8 b) internal returns (ebool) {
        return ebool.wrap(_eq(Types.EUINT8, euint8.unwrap(a), euint8.unwrap(b)));
    }
    
    function eq(euint32 a, euint32 b) internal returns (ebool) {
        return ebool.wrap(_eq(Types.EUINT32, euint32.unwrap(a), euint32.unwrap(b)));
    }
    
    function eq(euint64 a, euint64 b) internal returns (ebool) {
        return ebool.wrap(_eq(Types.EUINT64, euint64.unwrap(a), euint64.unwrap(b)));
    }

    // ============ Conditional Selection ============
    
    function select(ebool control, euint8 a, euint8 b) internal returns (euint8) {
        return euint8.wrap(_select(Types.EUINT8, ebool.unwrap(control), euint8.unwrap(a), euint8.unwrap(b)));
    }
    
    function select(ebool control, euint32 a, euint32 b) internal returns (euint32) {
        return euint32.wrap(_select(Types.EUINT32, ebool.unwrap(control), euint32.unwrap(a), euint32.unwrap(b)));
    }
    
    function select(ebool control, euint64 a, euint64 b) internal returns (euint64) {
        return euint64.wrap(_select(Types.EUINT64, ebool.unwrap(control), euint64.unwrap(a), euint64.unwrap(b)));
    }
    
    // ============ Boolean Operations (TFHE Gates) ============
    
    /// @notice Logical AND on encrypted booleans
    function and(ebool a, ebool b) internal returns (ebool) {
        return ebool.wrap(_and(ebool.unwrap(a), ebool.unwrap(b)));
    }
    
    /// @notice Logical OR on encrypted booleans
    function or(ebool a, ebool b) internal returns (ebool) {
        return ebool.wrap(_or(ebool.unwrap(a), ebool.unwrap(b)));
    }
    
    /// @notice Logical XOR on encrypted booleans
    function xor(ebool a, ebool b) internal returns (ebool) {
        return ebool.wrap(_xor(ebool.unwrap(a), ebool.unwrap(b)));
    }
    
    /// @notice Logical NOT on encrypted boolean
    function not(ebool a) internal returns (ebool) {
        return ebool.wrap(_not(ebool.unwrap(a)));
    }
    
    /// @notice Logical NAND on encrypted booleans
    function nand(ebool a, ebool b) internal returns (ebool) {
        return not(and(a, b));
    }
    
    /// @notice Logical NOR on encrypted booleans
    function nor(ebool a, ebool b) internal returns (ebool) {
        return not(or(a, b));
    }
    
    /// @notice Logical XNOR on encrypted booleans
    function xnor(ebool a, ebool b) internal returns (ebool) {
        return not(xor(a, b));
    }
    
    // ============ Bitwise Operations on Integers ============
    
    /// @notice Bitwise AND on encrypted integers
    function and(euint8 a, euint8 b) internal returns (euint8) {
        return euint8.wrap(_bitwiseAnd(Types.EUINT8, euint8.unwrap(a), euint8.unwrap(b)));
    }
    
    function and(euint32 a, euint32 b) internal returns (euint32) {
        return euint32.wrap(_bitwiseAnd(Types.EUINT32, euint32.unwrap(a), euint32.unwrap(b)));
    }
    
    function and(euint64 a, euint64 b) internal returns (euint64) {
        return euint64.wrap(_bitwiseAnd(Types.EUINT64, euint64.unwrap(a), euint64.unwrap(b)));
    }
    
    /// @notice Bitwise OR on encrypted integers
    function or(euint8 a, euint8 b) internal returns (euint8) {
        return euint8.wrap(_bitwiseOr(Types.EUINT8, euint8.unwrap(a), euint8.unwrap(b)));
    }
    
    function or(euint32 a, euint32 b) internal returns (euint32) {
        return euint32.wrap(_bitwiseOr(Types.EUINT32, euint32.unwrap(a), euint32.unwrap(b)));
    }
    
    function or(euint64 a, euint64 b) internal returns (euint64) {
        return euint64.wrap(_bitwiseOr(Types.EUINT64, euint64.unwrap(a), euint64.unwrap(b)));
    }
    
    /// @notice Bitwise XOR on encrypted integers
    function xor(euint8 a, euint8 b) internal returns (euint8) {
        return euint8.wrap(_bitwiseXor(Types.EUINT8, euint8.unwrap(a), euint8.unwrap(b)));
    }
    
    function xor(euint32 a, euint32 b) internal returns (euint32) {
        return euint32.wrap(_bitwiseXor(Types.EUINT32, euint32.unwrap(a), euint32.unwrap(b)));
    }
    
    function xor(euint64 a, euint64 b) internal returns (euint64) {
        return euint64.wrap(_bitwiseXor(Types.EUINT64, euint64.unwrap(a), euint64.unwrap(b)));
    }
    
    /// @notice Bitwise NOT on encrypted integers
    function not(euint8 a) internal returns (euint8) {
        return euint8.wrap(_bitwiseNot(Types.EUINT8, euint8.unwrap(a)));
    }
    
    function not(euint32 a) internal returns (euint32) {
        return euint32.wrap(_bitwiseNot(Types.EUINT32, euint32.unwrap(a)));
    }
    
    function not(euint64 a) internal returns (euint64) {
        return euint64.wrap(_bitwiseNot(Types.EUINT64, euint64.unwrap(a)));
    }
    
    // ============ Shift Operations ============
    
    function shl(euint8 a, euint8 bits) internal returns (euint8) {
        return euint8.wrap(_shl(Types.EUINT8, euint8.unwrap(a), euint8.unwrap(bits)));
    }
    
    function shl(euint32 a, euint32 bits) internal returns (euint32) {
        return euint32.wrap(_shl(Types.EUINT32, euint32.unwrap(a), euint32.unwrap(bits)));
    }
    
    function shr(euint8 a, euint8 bits) internal returns (euint8) {
        return euint8.wrap(_shr(Types.EUINT8, euint8.unwrap(a), euint8.unwrap(bits)));
    }
    
    function shr(euint32 a, euint32 bits) internal returns (euint32) {
        return euint32.wrap(_shr(Types.EUINT32, euint32.unwrap(a), euint32.unwrap(bits)));
    }
    
    // ============ Min/Max Operations ============
    
    function min(euint8 a, euint8 b) internal returns (euint8) {
        return select(lt(a, b), a, b);
    }
    
    function min(euint32 a, euint32 b) internal returns (euint32) {
        return select(lt(a, b), a, b);
    }
    
    function max(euint8 a, euint8 b) internal returns (euint8) {
        return select(gt(a, b), a, b);
    }
    
    function max(euint32 a, euint32 b) internal returns (euint32) {
        return select(gt(a, b), a, b);
    }

    // ============ Decryption ============
    
    function decrypt(ebool ct) internal returns (bool) {
        return _decrypt(Types.EBOOL, ebool.unwrap(ct)) != 0;
    }
    
    function decrypt(euint8 ct) internal returns (uint8) {
        return uint8(_decrypt(Types.EUINT8, euint8.unwrap(ct)));
    }
    
    function decrypt(euint16 ct) internal returns (uint16) {
        return uint16(_decrypt(Types.EUINT16, euint16.unwrap(ct)));
    }
    
    function decrypt(euint32 ct) internal returns (uint32) {
        return uint32(_decrypt(Types.EUINT32, euint32.unwrap(ct)));
    }
    
    function decrypt(euint64 ct) internal returns (uint64) {
        return uint64(_decrypt(Types.EUINT64, euint64.unwrap(ct)));
    }

    // ============ Random Number Generation ============
    
    function randomEuint8() internal returns (euint8) {
        return euint8.wrap(_random(Types.EUINT8, 0));
    }
    
    function randomEuint32() internal returns (euint32) {
        return euint32.wrap(_random(Types.EUINT32, 0));
    }
    
    function randomEuint64() internal returns (euint64) {
        return euint64.wrap(_random(Types.EUINT64, 0));
    }

    // ============ Internal Precompile Calls ============
    
    function _toBytes(uint256 x) private pure returns (bytes memory b) {
        b = new bytes(32);
        assembly { mstore(add(b, 32), x) }
    }
    
    function _getValue(bytes memory a) private pure returns (uint256 value) {
        assembly { value := mload(add(a, 0x20)) }
    }
    
    function _trivialEncrypt(uint256 value, uint8 toType, int32 securityZone) private returns (uint256) {
        bytes memory output = FheOps(Precompile.FHE).trivialEncrypt(_toBytes(value), toType, securityZone);
        return _getValue(output);
    }
    
    function _verify(bytes memory input, uint8 toType, int32 securityZone) private returns (uint256) {
        bytes memory output = FheOps(Precompile.FHE).verify(toType, input, securityZone);
        return _getValue(output);
    }
    
    function _add(uint8 utype, uint256 a, uint256 b) private returns (uint256) {
        bytes memory output = FheOps(Precompile.FHE).add(utype, _toBytes(a), _toBytes(b));
        return _getValue(output);
    }
    
    function _sub(uint8 utype, uint256 a, uint256 b) private returns (uint256) {
        bytes memory output = FheOps(Precompile.FHE).sub(utype, _toBytes(a), _toBytes(b));
        return _getValue(output);
    }
    
    function _mul(uint8 utype, uint256 a, uint256 b) private returns (uint256) {
        bytes memory output = FheOps(Precompile.FHE).mul(utype, _toBytes(a), _toBytes(b));
        return _getValue(output);
    }
    
    function _lt(uint8 utype, uint256 a, uint256 b) private returns (uint256) {
        bytes memory output = FheOps(Precompile.FHE).lt(utype, _toBytes(a), _toBytes(b));
        return _getValue(output);
    }
    
    function _gt(uint8 utype, uint256 a, uint256 b) private returns (uint256) {
        bytes memory output = FheOps(Precompile.FHE).gt(utype, _toBytes(a), _toBytes(b));
        return _getValue(output);
    }
    
    function _eq(uint8 utype, uint256 a, uint256 b) private returns (uint256) {
        bytes memory output = FheOps(Precompile.FHE).eq(utype, _toBytes(a), _toBytes(b));
        return _getValue(output);
    }
    
    function _select(uint8 utype, uint256 control, uint256 a, uint256 b) private returns (uint256) {
        bytes memory output = FheOps(Precompile.FHE).select(utype, _toBytes(control), _toBytes(a), _toBytes(b));
        return _getValue(output);
    }
    
    function _decrypt(uint8 utype, uint256 ct) private returns (uint256) {
        return FheOps(Precompile.FHE).decrypt(utype, _toBytes(ct), 0);
    }
    
    function _random(uint8 utype, int32 securityZone) private returns (uint256) {
        bytes memory output = FheOps(Precompile.FHE).random(utype, uint64(block.timestamp), securityZone);
        return _getValue(output);
    }
    
    // Boolean operations (TFHE gates)
    function _and(uint256 a, uint256 b) private returns (uint256) {
        bytes memory output = FheOps(Precompile.FHE).and(Types.EBOOL, _toBytes(a), _toBytes(b));
        return _getValue(output);
    }
    
    function _or(uint256 a, uint256 b) private returns (uint256) {
        bytes memory output = FheOps(Precompile.FHE).or(Types.EBOOL, _toBytes(a), _toBytes(b));
        return _getValue(output);
    }
    
    function _xor(uint256 a, uint256 b) private returns (uint256) {
        bytes memory output = FheOps(Precompile.FHE).xor(Types.EBOOL, _toBytes(a), _toBytes(b));
        return _getValue(output);
    }
    
    function _not(uint256 a) private returns (uint256) {
        bytes memory output = FheOps(Precompile.FHE).not(Types.EBOOL, _toBytes(a));
        return _getValue(output);
    }
    
    // Bitwise operations on integers
    function _bitwiseAnd(uint8 utype, uint256 a, uint256 b) private returns (uint256) {
        bytes memory output = FheOps(Precompile.FHE).and(utype, _toBytes(a), _toBytes(b));
        return _getValue(output);
    }
    
    function _bitwiseOr(uint8 utype, uint256 a, uint256 b) private returns (uint256) {
        bytes memory output = FheOps(Precompile.FHE).or(utype, _toBytes(a), _toBytes(b));
        return _getValue(output);
    }
    
    function _bitwiseXor(uint8 utype, uint256 a, uint256 b) private returns (uint256) {
        bytes memory output = FheOps(Precompile.FHE).xor(utype, _toBytes(a), _toBytes(b));
        return _getValue(output);
    }
    
    function _bitwiseNot(uint8 utype, uint256 a) private returns (uint256) {
        bytes memory output = FheOps(Precompile.FHE).not(utype, _toBytes(a));
        return _getValue(output);
    }
    
    // Shift operations
    function _shl(uint8 utype, uint256 a, uint256 bits) private returns (uint256) {
        bytes memory output = FheOps(Precompile.FHE).shl(utype, _toBytes(a), _toBytes(bits));
        return _getValue(output);
    }
    
    function _shr(uint8 utype, uint256 a, uint256 bits) private returns (uint256) {
        bytes memory output = FheOps(Precompile.FHE).shr(utype, _toBytes(a), _toBytes(bits));
        return _getValue(output);
    }
}
