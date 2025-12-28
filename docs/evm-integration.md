# FHE-EVM Integration Architecture

## Overview

This document outlines the architecture for integrating FHE with Lux EVM using the permissively-licensed OpenFHE (C++) and Lattice (Go) libraries.

## Lux FHE Stack

| Component | Implementation | License |
|-----------|----------------|---------|
| FHE Library | OpenFHE (C++) | BSD-2-Clause |
| Go Bindings | luxfi/fhe/go | BSD-2-Clause |
| Lattice HE | luxfi/lattice | Apache-2.0 |
| EVM Integration | Precompile + T-Chain | - |
| Threshold Decrypt | Validator Set via MPC | - |

### Key Advantages

- **Permissive Licensing**: BSD-2-Clause allows commercial use without restrictions
- **No Vendor Lock-in**: Pure open-source stack, no proprietary coprocessors
- **Multi-Scheme Support**: TFHE, FHEW, CKKS, BGV/BFV all available
- **Go-Native Option**: Lattice library provides pure Go HE for microservices

## Core Components

### 1. Solidity Library (`FHE.sol`)

Encrypted types wrapping uint256 handles:

```solidity
type ebool is uint256;
type euint8 is uint256;
type euint16 is uint256;
type euint32 is uint256;
type euint64 is uint256;
type euint128 is uint256;
type euint256 is uint256;
type eaddress is uint256;
```

Operations via precompile calls:
- Arithmetic: `add`, `sub`, `mul`, `div`, `rem`
- Comparison: `lt`, `lte`, `gt`, `gte`, `eq`, `ne`, `min`, `max`
- Bitwise: `and`, `or`, `xor`, `not`, `shl`, `shr`, `rol`, `ror`
- Control: `select` (ternary), `req` (require on encrypted)
- Utility: `cast`, `trivialEncrypt`, `decrypt`, `sealOutput`

### 2. FHE Precompile (`0x80`)

Interface at address 128:

```go
type FHEPrecompile struct {
    ctx *tfhe.Context
}

func (p *FHEPrecompile) Run(input []byte) ([]byte, error) {
    opcode := input[0]
    switch opcode {
    case OP_ADD:
        return p.add(input[1:])
    case OP_VERIFY:
        return p.verify(input[1:])
    case OP_DECRYPT:
        return p.decrypt(input[1:])
    // ... etc
    }
}
```

### 3. Ciphertext Storage

Handles are uint256 referencing stored ciphertexts:
- On-chain: Only store handles (32 bytes)
- Off-chain: Store actual ciphertexts in coprocessor/DB
- Mapping: `handle -> ciphertext` in state trie or separate store

### 4. Coprocessor Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   EVM / Node    │────▶│   Coprocessor    │────▶│   OpenFHE       │
│  (precompiles)  │     │   (Go service)   │     │   (C++ lib)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                        │                        │
        │  emit FHE event        │  execute TFHE          │
        │                        │                        │
        ▼                        ▼                        ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   State Trie    │     │   Result Queue   │     │   Keys Store    │
│   (handles)     │     │   (pending ops)  │     │   (sk shares)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

Two execution models:

**A. Synchronous (simpler, slower)**
- Precompile blocks until FHE operation completes
- ~10-100ms per operation
- Suitable for low-volume chains

**B. Asynchronous (production)**
- Precompile returns immediately with handle
- Actual computation queued to coprocessor
- Results written back via callback
- Higher throughput, complex state management

### 5. Threshold Decryption

For decrypt operations, use threshold FHE across validators:

```
┌─────────────────────────────────────────────┐
│                Decrypt Request              │
└─────────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ Val 1   │   │ Val 2   │   │ Val 3   │
   │ Share 1 │   │ Share 2 │   │ Share 3 │
   └─────────┘   └─────────┘   └─────────┘
        │             │             │
        └─────────────┼─────────────┘
                      ▼
          ┌───────────────────────┐
          │  Combine (t-of-n)     │
          │  Plaintext Result     │
          └───────────────────────┘
```

Using Lattice's multiparty package:

```go
import "github.com/luxfi/lattice/multiparty"

// Setup 3-of-5 threshold
params := ckks.NewParametersFromLiteral(ckks.PN14QP438)
crs := multiparty.NewCRS(params.Parameters)

// Each validator generates key share
shares := make([]*multiparty.KeyShare, n)
for i := range validators {
    shares[i] = multiparty.GenKeyShare(crs, params)
}

// Combine for decryption
plaintext := multiparty.ThresholdDecrypt(ciphertext, shares[:t])
```

### 6. Key Management

**Network Key Pair**:
- Public key: Available to all for encryption
- Secret key: Distributed as shares across validators

**User Sealing Keys**:
- User provides public key for re-encryption
- `sealOutput` re-encrypts under user's key
- Only user can decrypt sealed output

## Gas Costs

Estimated gas for FHE operations:

| Operation | euint8 | euint32 | euint64 | euint256 |
|-----------|--------|---------|---------|----------|
| add       | 50k    | 60k     | 80k     | 150k     |
| mul       | 100k   | 150k    | 250k    | 500k     |
| lt/gt/eq  | 80k    | 100k    | 150k    | 300k     |
| select    | 60k    | 80k     | 120k    | 250k     |
| decrypt   | 200k   | 200k    | 200k    | 200k     |

## T-Chain Integration

The T-Chain (Threshold Chain) provides infrastructure for:

1. **Key Generation Ceremonies**: Distributed key gen across validators
2. **Decryption Coordination**: Aggregate partial decryptions
3. **Key Rotation**: Periodic resharing without revealing secret
4. **Access Control**: On-chain ACLs for decrypt permissions

## Implementation Plan

### Phase 1: Basic Integration
- [ ] FHE precompile with synchronous execution
- [ ] OpenFHE Go bindings integration
- [ ] Handle storage in state trie
- [ ] Basic Solidity library

### Phase 2: Threshold Decryption
- [ ] Lattice multiparty integration
- [ ] Validator key share management
- [ ] T-Chain coordination protocol
- [ ] Secure key ceremonies

### Phase 3: Async Coprocessor
- [ ] Event-based FHE execution
- [ ] Result callback mechanism
- [ ] Parallel operation batching
- [ ] GPU acceleration (optional)

### Phase 4: Production Hardening
- [ ] Comprehensive gas metering
- [ ] Ciphertext garbage collection
- [ ] Key rotation protocol
- [ ] Security audit

## References

- [OpenFHE Documentation](https://openfhe-development.readthedocs.io/)
- [Lux FHE Library](https://github.com/luxfi/fhe)
- [Lux Lattice Library](https://github.com/luxfi/lattice)
- [TFHE Original Paper](https://eprint.iacr.org/2018/421)
- [CKKS Paper](https://eprint.iacr.org/2016/421)
