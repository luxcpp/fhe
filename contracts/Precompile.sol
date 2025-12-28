// SPDX-License-Identifier: BSD-3-Clause
pragma solidity ^0.8.20;

/// @title FHE Precompile Address
library Precompile {
    address internal constant FHE = address(128);
}

/// @title FHE Operations Interface
/// @notice Interface to the FHE precompile at address 0x80
/// @dev Implemented by the Lux EVM as a stateful precompile
interface FheOps {
    /// @notice Verify and import an encrypted input
    function verify(uint8 utype, bytes calldata input, int32 securityZone) external returns (bytes memory);
    
    /// @notice Encrypt a plaintext value (trivial encryption)
    function trivialEncrypt(bytes calldata input, uint8 toType, int32 securityZone) external returns (bytes memory);
    
    /// @notice Decrypt a ciphertext (requires threshold decryption)
    function decrypt(uint8 utype, bytes calldata input, uint256 defaultValue) external returns (uint256);
    
    /// @notice Seal output for a specific public key
    function sealOutput(uint8 utype, bytes calldata ctHash, bytes calldata pk) external returns (string memory);
    
    /// @notice Get network public key for encryption
    function getNetworkPublicKey(int32 securityZone) external view returns (bytes memory);
    
    // Arithmetic operations
    function add(uint8 utype, bytes calldata lhs, bytes calldata rhs) external returns (bytes memory);
    function sub(uint8 utype, bytes calldata lhs, bytes calldata rhs) external returns (bytes memory);
    function mul(uint8 utype, bytes calldata lhs, bytes calldata rhs) external returns (bytes memory);
    function div(uint8 utype, bytes calldata lhs, bytes calldata rhs) external returns (bytes memory);
    function rem(uint8 utype, bytes calldata lhs, bytes calldata rhs) external returns (bytes memory);
    
    // Comparison operations
    function lt(uint8 utype, bytes calldata lhs, bytes calldata rhs) external returns (bytes memory);
    function lte(uint8 utype, bytes calldata lhs, bytes calldata rhs) external returns (bytes memory);
    function gt(uint8 utype, bytes calldata lhs, bytes calldata rhs) external returns (bytes memory);
    function gte(uint8 utype, bytes calldata lhs, bytes calldata rhs) external returns (bytes memory);
    function eq(uint8 utype, bytes calldata lhs, bytes calldata rhs) external returns (bytes memory);
    function ne(uint8 utype, bytes calldata lhs, bytes calldata rhs) external returns (bytes memory);
    function min(uint8 utype, bytes calldata lhs, bytes calldata rhs) external returns (bytes memory);
    function max(uint8 utype, bytes calldata lhs, bytes calldata rhs) external returns (bytes memory);
    
    // Bitwise operations
    function and(uint8 utype, bytes calldata lhs, bytes calldata rhs) external returns (bytes memory);
    function or(uint8 utype, bytes calldata lhs, bytes calldata rhs) external returns (bytes memory);
    function xor(uint8 utype, bytes calldata lhs, bytes calldata rhs) external returns (bytes memory);
    function not(uint8 utype, bytes calldata value) external returns (bytes memory);
    function shl(uint8 utype, bytes calldata lhs, bytes calldata rhs) external returns (bytes memory);
    function shr(uint8 utype, bytes calldata lhs, bytes calldata rhs) external returns (bytes memory);
    function rol(uint8 utype, bytes calldata lhs, bytes calldata rhs) external returns (bytes memory);
    function ror(uint8 utype, bytes calldata lhs, bytes calldata rhs) external returns (bytes memory);
    
    // Control flow
    function select(uint8 utype, bytes calldata control, bytes calldata ifTrue, bytes calldata ifFalse) external returns (bytes memory);
    function req(uint8 utype, bytes calldata input) external returns (bytes memory);
    
    // Type conversion
    function cast(uint8 utype, bytes calldata input, uint8 toType) external returns (bytes memory);
    
    // Random number generation
    function random(uint8 utype, uint64 seed, int32 securityZone) external returns (bytes memory);
    
    // Utility
    function square(uint8 utype, bytes calldata value) external returns (bytes memory);
}
