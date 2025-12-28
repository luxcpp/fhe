# Lux FHE - Fully Homomorphic Encryption

## Overview

This is a fork of [OpenFHE](https://github.com/openfheorg/openfhe-development) providing permissively-licensed FHE for the Lux ecosystem. Licensed under BSD-3-Clause.

**Key advantages:**
- **BSD-3-Clause license** - Permissive, no patent restrictions
- **TFHE/CGGI support** - Essential for blockchain (fast ~10ms bootstrapping)
- **Academic origins** - Algorithms from peer-reviewed research
- **Production-ready** - Used by DARPA DPRIVE program
- **Multi-scheme** - TFHE, FHEW, CKKS, BGV, BFV all supported

## Technology

### Encryption Schemes

| Scheme | Use Case | Performance |
|--------|----------|-------------|
| TFHE/CGGI | Boolean circuits | ~10ms bootstrapping |
| FHEW | Binary operations | Functional bootstrapping |
| CKKS | Real numbers | Approximate arithmetic |
| BGV | Integers | Exact arithmetic (modular) |
| BFV | Integers | Exact arithmetic (scaled) |

### Key Concepts

- **Bootstrapping**: Noise reduction allowing unlimited computation depth
- **LWE/RLWE**: Learning With Errors - core hardness assumption
- **Threshold FHE**: Distributed key generation and decryption
- **Functional Bootstrapping**: Evaluate arbitrary functions during bootstrap

## Directory Structure

```
fhe/
├── src/
│   ├── core/           # Math primitives, lattice operations
│   ├── binfhe/         # TFHE/FHEW binary FHE
│   └── pke/            # CKKS/BGV/BFV public key encryption
├── go/
│   ├── tfhe/           # Go bindings for TFHE
│   │   ├── context.go  # CGO bindings
│   │   ├── bridge.cpp  # C++ bridge
│   │   └── compare.go  # Integer comparisons
│   ├── ckks/           # CKKS bindings (placeholder)
│   └── threshold/      # Threshold FHE bindings (placeholder)
├── build/
│   ├── lib/            # Compiled libraries
│   │   ├── libOPENFHEcore.dylib
│   │   ├── libOPENFHEbinfhe.dylib
│   │   └── libOPENFHEpke.dylib
│   └── bin/examples/   # Example binaries
└── third-party/        # Dependencies (cereal, google-test)
```

## Build

```bash
# Configure
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_UNITTESTS=OFF \
         -DBUILD_BENCHMARKS=OFF

# Build
make -j$(sysctl -n hw.ncpu)

# Libraries output to build/lib/
```

## Go Bindings

CGO bindings for Lux node integration:

```go
import "github.com/luxfi/fhe/go/tfhe"

// Create context
ctx := tfhe.NewContext(tfhe.STD128)
defer ctx.Close()

// Generate keys
sk := ctx.KeyGen()
ctx.BootstrapKeyGen(sk)

// Encrypt and compute
ct1 := ctx.Encrypt(sk, true)
ct2 := ctx.Encrypt(sk, false)
result := ctx.AND(ct1, ct2)

// Decrypt
plaintext := ctx.Decrypt(sk, result)
```

## Key Files

| File | Purpose |
|------|---------|
| `src/binfhe/lib/binfhecontext.cpp` | Main TFHE context implementation |
| `src/binfhe/lib/lwe-pke.cpp` | LWE encryption/decryption |
| `src/binfhe/lib/rgsw-acc-cggi.cpp` | CGGI accumulator (fast bootstrapping) |
| `src/binfhe/lib/rgsw-acc-lmkcdey.cpp` | LMKCDEY variant |
| `go/tfhe/context.go` | Go CGO bindings |
| `go/tfhe/bridge.cpp` | C++ bridge code |

## Parameters

### Security Levels

| Set | LWE n | RLWE N | Security |
|-----|-------|--------|----------|
| STD128 | 512 | 1024 | 128-bit |
| STD128_AP | 512 | 1024 | 128-bit (AP variant) |
| STD128_LMKCDEY | 512 | 2048 | 128-bit (fast) |
| STD192 | 2048 | 4096 | 192-bit |

### Parameter Selection

- **STD128_LMKCDEY**: Fastest for EVM use (recommended)
- **STD128**: Standard CGGI, balanced
- **STD192**: Higher security for sensitive applications

## EVM Integration

### Architecture

The fhEVM integration consists of:

1. **FHE.sol** - Solidity library with encrypted types (ebool, euint8...euint256)
2. **Precompile** - EVM precompile at address 128 handling FHE ops
3. **Go bindings** - CGO bridge to OpenFHE C++ library
4. **Coprocessor** - Off-chain FHE execution (async model)
5. **Threshold decrypt** - Validator-based distributed decryption

### Licensing Advantages

| Aspect | Lux FHE |
|--------|---------|
| License | BSD-3-Clause |
| Commercial Use | ✅ Unrestricted |
| Patent Risk | Low (academic) |
| Language | C++ |
| Integration | CGO → Go |

## Benchmarks

### TFHE Results (Apple M-series ARM64)

| Operation | Lux FHE (LMKCDEY) | Lux FHE (GINX) |
|-----------|-------------------|----------------|
| Key Gen | ~1500 ms | ~1500 ms |
| Encrypt | ~0.1 ms | ~0.1 ms |
| AND | ~13 ms | ~50 ms |
| OR | ~13 ms | ~50 ms |
| XOR | ~13 ms | ~50 ms |
| NAND | ~13 ms | ~50 ms |
| MUX | ~26 ms | ~100 ms |

**Recommendation**: Use LMKCDEY for best performance.

### CKKS Results (Apple M-series ARM64)

| Operation | Lux Lattice (Go) | Lux Lattice (Parallel) | Notes |
|-----------|------------------|------------------------|-------|
| Encode | 2.15 ms | - | LogN=14, 8K slots |
| Decode | 13.04 ms | - | |
| Add Ciphertext | 0.12 ms | 0.02 ms | ~6x speedup |
| Mul Ciphertext | 0.66 ms | 0.12 ms | ~5.5x speedup |
| MulRelin | 18.95 ms | 2.85 ms | ~6.6x speedup |
| Rescale | 2.20 ms | 0.34 ms | ~6.5x speedup |
| Rotate | 17.55 ms | 2.79 ms | ~6.3x speedup |

**Pure Go Lattice Library**: `~/work/lux/lattice`
- Full CKKS implementation with parallel evaluator
- Apache 2.0 license
- No CGO required - pure Go

### Scheme Selection Guide

| Use Case | Recommended Library | Notes |
|----------|---------------------|-------|
| Boolean circuits (fhEVM) | Lux FHE (OpenFHE) | TFHE/CGGI via CGO |
| Real number arithmetic | Lux Lattice | Pure Go CKKS |
| Integer modular arithmetic | Lux FHE or Lattice | BGV scheme |
| Threshold decryption | Lux Lattice | Built-in multiparty |

**Recommendation**:
- Use **Lux Lattice** for CKKS workloads (pure Go, no CGO overhead)
- Use **Lux FHE** for TFHE/boolean circuits (via CGO bindings)

## GPU Coprocessor Architecture

The GPU coprocessor enables high-throughput FHE operations for fhEVM workloads. Key components:

### Directory Structure (New Extensions)

```
src/binfhe/
├── include/
│   ├── batch/          # Batch APIs for GPU throughput
│   │   └── binfhe-batch.h
│   ├── radix/          # Radix integer arithmetic
│   │   └── radix.h
│   └── backend/        # GPU backend interfaces
│       └── backend.h
└── lib/
    ├── batch/          # Batch implementations
    ├── radix/          # Radix implementations
    └── backend/        # MLX/CUDA backends
```

### Radix Integer Types

Support for encrypted integers larger than the modulus:

```cpp
// euint256 = 32 x euint8 limbs
struct RadixCiphertext {
    std::vector<LWECiphertext> limbs;
    uint32_t bits_per_limb;  // 8 for euint8 limbs
    uint32_t total_bits;     // 256 for euint256
};

// Operations
RadixCiphertext add(const RadixCiphertext& a, const RadixCiphertext& b);
RadixCiphertext sub(const RadixCiphertext& a, const RadixCiphertext& b);
LWECiphertext lt(const RadixCiphertext& a, const RadixCiphertext& b);
```

### Batch APIs

For GPU throughput, batch APIs minimize kernel launch overhead:

```cpp
void BootstrapBatch(
    const std::vector<LWECiphertext>& ct_in,
    std::vector<LWECiphertext>& ct_out
);

void EvalFuncBatch(
    const std::vector<LWECiphertext>& ct_in,
    const LUT& lut,
    std::vector<LWECiphertext>& ct_out
);
```

### Execution Models

**Synchronous (Simple)**:
- Precompile blocks until FHE operation completes
- ~10-100ms per operation
- Suitable for low-volume chains

**Asynchronous (Production)**:
- Precompile returns immediately with handle
- Actual computation queued to coprocessor
- Results written back via callback
- Higher throughput, complex state management

## Threshold Integration

T-Chain provides distributed decryption:

```go
import "github.com/luxfi/lattice/multiparty"

// Setup 3-of-5 threshold
params := ckks.NewParametersFromLiteral(ckks.PN14QP438)
crs := multiparty.NewCRS(params.Parameters)

// Each validator generates key share
shares := make([]*multiparty.SecretShare, n)
for i := range validators {
    shares[i] = multiparty.GenSecretShare(crs, i)
}

// Combine partial decryptions
plaintext := multiparty.ThresholdDecrypt(ciphertext, partialDecrypts)
```

## Troubleshooting

### Build Issues

**Missing cereal**:
```bash
git submodule update --init --recursive
```

**macOS ARM64**:
```bash
cmake .. -DCMAKE_OSX_ARCHITECTURES=arm64
```

### Runtime Issues

**CGO linking errors**: Ensure `CGO_LDFLAGS` points to build/lib/

**Slow performance**: Use LMKCDEY parameter set for faster bootstrapping

## References

- [OpenFHE Documentation](https://openfhe-development.readthedocs.io/)
- [TFHE Paper](https://eprint.iacr.org/2018/421)
- [CKKS Paper](https://eprint.iacr.org/2016/421)
- [LMKCDEY Paper](https://eprint.iacr.org/2022/198)
- [Lux Lattice Library](https://github.com/luxfi/lattice)

---

*Last Updated: 2025-12-27*
