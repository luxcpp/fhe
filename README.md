# FHE - Fully Homomorphic Encryption

Computation on encrypted data without decryption.

## Overview

FHE enables privacy-preserving computation using only cryptographic operations.
This implementation supports:

- **TFHE/CGGI**: Fast boolean circuits with ~10ms bootstrapping
- **FHEW**: Binary operations with functional bootstrapping  
- **CKKS**: Approximate arithmetic on real numbers
- **BGV/BFV**: Exact arithmetic on integers
- **Threshold FHE**: Distributed decryption across parties

## Security Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FHE COMPUTATION                             │
│                                                                     │
│  Client                          │       Compute Node               │
│  ┌─────────────────────────┐     │      ┌──────────────────────┐   │
│  │  Encrypt locally        │     │      │  Compute on cipher   │   │
│  │  (holds secret key)     │────────────│  (no secret key!)    │   │
│  │                         │     │      │                      │   │
│  │  • Key generation       │     │      │  • Add ciphertexts   │   │
│  │  • Encryption           │     │      │  • Multiply          │   │
│  │  • Decryption           │     │      │  • Bootstrap         │   │
│  └─────────────────────────┘     │      └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

**Key insight**: Computation happens on ENCRYPTED data. 
The compute node never sees plaintext - security by math, not trust.

## Installation

### C++ Library
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

### Go
```bash
go get github.com/luxfi/fhe/go
```

## Quick Start

### TFHE: Boolean Gates
```go
import "github.com/luxfi/fhe/go/tfhe"

// Create context
ctx := tfhe.NewContext(tfhe.STD128)
defer ctx.Close()

// Generate keys
sk := ctx.KeyGen()
ctx.BootstrapKeyGen(sk)

// Encrypt bits
a := ctx.Encrypt(sk, true)
b := ctx.Encrypt(sk, false)

// Compute AND gate on encrypted data
result := ctx.AND(a, b)

// Decrypt
plain := ctx.Decrypt(sk, result) // false
```

### TFHE: Comparison
```go
// Encrypt 8-bit integers
x := ctx.EncryptInt8(sk, 42)
y := ctx.EncryptInt8(sk, 17)

// Compare encrypted values
greater := ctx.GreaterThan(x, y)

// Decrypt result
isGreater := ctx.Decrypt(sk, greater) // true
```

### Threshold Decryption
```go
import "github.com/luxfi/fhe/go/threshold"

// Setup 3-of-5 threshold
parties := threshold.Setup(3, 5)

// Each party generates partial key
shares := make([]*threshold.Share, 5)
for i, p := range parties {
    shares[i] = p.KeyGen()
}

// Combine into threshold public key
tpk := threshold.CombinePublic(shares)

// Encrypt with threshold key
ct := ctx.EncryptWithPK(tpk, message)

// Partial decryptions (any 3 parties)
partials := make([]*threshold.Partial, 3)
for i := 0; i < 3; i++ {
    partials[i] = parties[i].PartialDecrypt(ct)
}

// Combine partials
plain := threshold.Combine(partials)
```

## Directory Structure

```
fhe/
├── src/
│   ├── binfhe/      # TFHE/FHEW (boolean FHE)
│   ├── pke/         # BGV/BFV/CKKS (arithmetic FHE)
│   └── core/        # Math primitives
├── go/              # Go bindings
│   ├── tfhe/        # TFHE bindings
│   ├── ckks/        # CKKS bindings
│   └── threshold/   # Threshold decryption
├── benchmark/       # Performance tests
└── docs/            # Documentation
```

## Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| TFHE Bootstrap | ~10ms | Single gate |
| TFHE AND/OR/XOR | ~10ms | With bootstrap |
| CKKS Add | ~0.1ms | Leveled |
| CKKS Mult | ~1ms | Leveled |
| Threshold (3/5) | ~50ms | Partial combine |

*AMD EPYC 7763, single thread*

## Schemes

### TFHE (Boolean)
Best for: comparisons, conditionals, binary logic
```
Encrypt(bit) → Ciphertext
AND/OR/XOR/NOT → Ciphertext
Bootstrap → Refreshed Ciphertext
```

### CKKS (Approximate)
Best for: ML inference, floating-point computation
```
Encrypt([f64; N]) → Ciphertext
Add/Mult → Ciphertext  
Rescale → Lower noise
Bootstrap → Full refresh
```

### BGV/BFV (Exact)
Best for: integer arithmetic, modular computation
```
Encrypt([i64; N]) → Ciphertext
Add/Mult mod p → Ciphertext
```

## Building

```bash
# Dependencies
brew install cmake  # macOS
apt install cmake   # Linux

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_UNITTESTS=ON
make -j$(nproc)

# Test
ctest --output-on-failure

# Install
sudo make install
```

## Go Bindings

```bash
cd go
CGO_ENABLED=1 go build ./...
go test ./...
```

## License

BSD 2-Clause. See [LICENSE](LICENSE).

Based on [OpenFHE](https://github.com/openfheorg/openfhe-development).

## References

- [TFHE: Fast FHE over the Torus](https://eprint.iacr.org/2018/421)
- [OpenFHE Design Paper](https://eprint.iacr.org/2022/915)
- [CKKS Scheme](https://eprint.iacr.org/2016/421)
