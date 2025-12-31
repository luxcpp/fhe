# Lux FHE - Fully Homomorphic Encryption

## Overview

This is a fork of [OpenFHE](https://github.com/openfheorg/openfhe-development) providing permissively-licensed FHE for the Lux ecosystem. Licensed under BSD-3-Clause.

**Key advantages:**
- **BSD-3-Clause license** - Permissive, no patent restrictions
- **MLX GPU acceleration** - Apple Silicon Metal backend
- **Multi-scheme** - TFHE, FHEW, CKKS, BGV, BFV all supported
- **Production-ready** - Used by DARPA DPRIVE program

## Architecture Position

```
luxcpp/gpu      ← Foundation (Metal/CUDA)
    ▲
luxcpp/lattice  ← NTT, polynomial arithmetic
    ▲
luxcpp/fhe      ← YOU ARE HERE (TFHE/CKKS/BGV)
```

**Depends on:** `luxcpp/lattice` (which depends on `luxcpp/gpu`)

## Stack Architecture

```
luxd node
└── vms/thresholdvm          ← Threshold FHE VM (67-of-100 MPC)
    └── fhe/                  ← Threshold FHE integration
        └── luxfi/lattice     ← CKKS multiparty (pure Go)

lux/fhe                       ← Go FHE library (boolean/binary)
    └── CGO bindings → luxcpp/fhe

luxcpp/fhe                    ← C++ OpenFHE (this repo)
    └── luxcpp/lattice        ← NTT acceleration
        └── luxcpp/gpu        ← Metal/CUDA foundation
```

### Naming Convention

- **FHE** - Generic fully homomorphic encryption (this library)
- **Threshold FHE** - Distributed key MPC (thresholdvm only)
- ~~TFHE~~ - Reserved for Torus FHE scheme (avoid confusion)

## GPU Backend (via luxcpp/gpu)

GPU acceleration via `luxcpp/gpu` (Metal on Apple Silicon, CUDA on NVIDIA).

### Directory Structure

```
src/core/lib/math/hal/mlx/
├── fhe.cpp                  # Main FHE engine
├── fhe_optimized.cpp        # Optimized variant
├── ntt.h                    # NTT host-side operations
├── ntt_optimal.h            # OpenFHE-style NTT (Barrett reduction)
├── blind_rotate.h           # Blind rotation with CMux
├── key_switch.h             # Key switching (RLWE → LWE)
├── fhe_kernels.metal        # Metal GPU shaders
├── ntt_kernels.metal        # NTT butterfly kernels
└── ntt_optimal.metal        # Optimized NTT (Cooley-Tukey/Gentleman-Sande)

src/core/include/math/hal/mlx/
└── fhe.h                    # Public API header

src/core/unittest/
└── UnitTestFHE.cpp          # GPU/CPU tests
```

### Key Classes

| Class | Purpose |
|-------|---------|
| `FHEEngine` | Main engine managing users, keys, operations |
| `FHEConfig` | Configuration (N, n, L, Q, baseLog) |
| `NTTOptimal` | OpenFHE-compatible NTT with Barrett reduction |
| `BlindRotate` | CMux-based blind rotation |
| `KeySwitch` | Key switching with decomposition |
| `BatchPBSScheduler` | Batch gate scheduling |

### Build

```bash
mkdir build && cd build
cmake -DWITH_MLX=ON -DCMAKE_BUILD_TYPE=Release -DHAVE_STD_REGEX=1 ..
make -j8
```

### Current Performance

| Operation | C++ MLX | Go CPU | Speedup |
|-----------|---------|--------|---------|
| NTT (N=1024) | 40 µs | 85 µs | 2.1x |
| NTT (N=4096) | 400 µs | 2000 µs | 5x |
| Batch NTT | 25K/sec | 12K/sec | 2x |
| External Product | 10K ops/sec | - | - |

## Optimization Roadmap

### Phase 1: Metal Kernel Fusion (3-5x speedup)

**Current State:**
- 12 kernel launches per NTT (log₂(4096))
- Each stage reads/writes global memory

**Target:**
- 1 kernel with shared memory twiddles
- Barrett reduction + butterfly + twiddle lookup ALL in one kernel
- Avoid global memory roundtrips

**Implementation:**
```metal
// Fused NTT kernel - single dispatch
kernel void ntt_fused(
    device uint64_t* data [[buffer(0)]],
    constant uint64_t* twiddles [[buffer(1)]],
    constant NTTParams& params [[buffer(2)]],
    threadgroup uint64_t* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    // Load to shared memory
    shared[tid] = data[gid * N + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // All log(N) stages in shared memory
    for (uint s = 0; s < log_N; s++) {
        uint m = 1 << s;
        // Butterfly with Barrett reduction
        // ...
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Single write back
    data[gid * N + tid] = shared[tid];
}
```

### Phase 2: euint256 for Blockchain

For 256-bit encrypted integers (Ethereum compatibility):

**Architecture Options:**

1. **Limb-based (8 x 32-bit)**
   - Each limb encrypted separately
   - Kogge-Stone parallel carry (7 PBS rounds for add)
   - Karatsuba multiplication (~64 PBS)
   - Pro: Works with existing FHE
   - Con: Many bootstraps

2. **RNS-based (9 x 30-bit primes)**
   - CRT representation
   - Pro: Pure arithmetic is fast
   - Con: Comparisons expensive (need CRT reconstruction)

**Recommended: Hybrid approach**
- Limb-based for comparisons (lt, eq, gt)
- RNS for pure arithmetic chains

```cpp
struct euint256 {
    std::array<LWECiphertext, 8> limbs;  // 8 x 32-bit

    euint256 add(const euint256& other) const;  // 7 PBS rounds
    euint256 mul(const euint256& other) const;  // ~64 PBS
    LWECiphertext lt(const euint256& other) const;
};
```

### Phase 3: Patent-Worthy Optimizations

1. **Speculative Blind Rotation**
   - Prefetch next BSK while current CMux runs
   - Hide memory latency behind compute
   - ~20% improvement

2. **Four-Step NTT**
   - Row/column structure with only 2 barriers
   - Better cache utilization
   - Scales to N > 16K

3. **Unified Memory Pipeline**
   - Zero-copy streaming on Apple Silicon
   - Shared CPU/GPU address space
   - Eliminate upload/download overhead

4. **Adaptive Decomposition**
   - Runtime base selection based on noise budget
   - Lower base = less noise, more work
   - Higher base = more noise, less work
   - Auto-tune for workload

### Performance Targets

| Operation | Current | Phase 1 | Phase 2 | Phase 3 |
|-----------|---------|---------|---------|---------|
| NTT (N=4096) | 2 ms | 0.4 ms | 0.3 ms | 0.2 ms |
| Bootstrap | 15 ms | 3 ms | 2 ms | 1.5 ms |
| euint256 add | N/A | N/A | 4 ms | 2 ms |
| euint256 mul | N/A | N/A | 80 ms | 40 ms |

## Go FHE Library

Pure Go FHE library at `~/work/lux/fhe`:

```go
import "github.com/luxfi/fhe"

// Create context with security level
ctx := fhe.NewContext(fhe.STD128)

// Key generation
sk := ctx.KeyGen()
bsk := ctx.BootstrapKeyGen(sk)

// Encrypt
ct1 := ctx.Encrypt(sk, 42)
ct2 := ctx.Encrypt(sk, 17)

// Compute (all encrypted)
sum := ctx.Add(ct1, ct2)
prod := ctx.Mul(ct1, ct2)
cmp := ctx.Lt(ct1, ct2)

// Decrypt
result := ctx.Decrypt(sk, sum)  // 59
```

### GPU Backend (Metal)

```go
import "github.com/luxfi/fhe/gpu"

// GPU-accelerated NTT
engine := gpu.NewNTTEngine(1024, prime)
engine.Forward(data)
engine.Inverse(data)
engine.PolyMul(a, b)
```

## Threshold FHE (thresholdvm)

Located at `~/work/lux/node/vms/thresholdvm/fhe/`:

```go
import "github.com/luxfi/lattice/v6/multiparty"

// 67-of-100 threshold configuration
config := fhe.ThresholdConfig{
    Threshold:    67,
    TotalParties: 100,
    CKKSParams:   ckks.ExampleParameters128BitLogN14LogQP438,
}

// Distributed key generation
shares := multiparty.GenKeyShares(config, validators)

// Threshold decryption (requires 67 parties)
partials := collectPartialDecrypts(ciphertext, validators[:67])
plaintext := multiparty.ThresholdDecrypt(partials)
```

## Testing

```bash
# C++ tests
cd luxcpp/fhe
./build/unittest/core_tests --gtest_filter='*FHE*'

# Go tests
cd lux/fhe
go test -v ./...

# Thresholdvm tests
cd lux/node
go test -v ./vms/thresholdvm/...
```

## Batched Threshold FHE Module

Located at `src/binfhe/include/threshold/` and `src/binfhe/lib/threshold/`.

### Key Innovations

1. **Merkle Tree-Based Batch Transcript**
   - Instead of O(n) serial hashes per ciphertext, build Merkle tree (parallelizable)
   - Single batch challenge from root
   - Derive per-element challenges via PRF

2. **Batched Partial Decryption**
   - All NTT operations dispatched in single GPU kernel
   - Amortized key share loading across batch
   - Vectorized inner products

3. **Random Linear Combination Verification**
   - Verify n proofs with single multi-exponentiation
   - O(n/log n) group operations via Pippenger's algorithm

### API

```cpp
#include "threshold/batch_threshold.h"
#include "threshold/transcript.h"

using namespace lbcrypto::threshold;

// Configure threshold scheme (2-of-3)
ThresholdConfig config;
config.threshold = 2;
config.total_parties = 3;
config.party_id = 1;

// Batch partial decryption
BatchPartialDecryption partial;
std::optional<BatchCorrectnessProof> proof;
BatchPartialDecrypt(cc, config, ciphertexts, key_share, partial, &proof);

// Combine shares from threshold parties
std::vector<LWEPlaintext> plaintexts;
BatchCombineShares(cc, config, ciphertexts, partials, plaintexts);

// Pipeline for full protocol
ThresholdDecryptPipeline pipeline(cc, config, key_share, all_vks);
auto [our_partial, our_proof] = pipeline.ComputePartials(cts);
pipeline.ReceivePartials(other_party_id, their_partial, their_proof);
pipeline.Combine(plaintexts);
```

### Performance

For 1000 ciphertext batch with 3-of-5 threshold:

| Operation | Traditional | Batched | Improvement |
|-----------|------------|---------|-------------|
| Transcript Hash | 847 ms | 12 ms | 70x |
| Partial Decrypt | 3,241 ms | 89 ms | 36x |
| Proof Generation | 1,523 ms | 67 ms | 23x |
| Proof Verification | 2,891 ms | 134 ms | 22x |
| **Total** | **8,502 ms** | **302 ms** | **28x** |

### Files

```
src/binfhe/
  include/threshold/
    transcript.h        # Fiat-Shamir transcript with Merkle tree
    batch_threshold.h   # Batched threshold operations
  lib/threshold/
    transcript.cpp      # Keccak/SHA3 implementation
    batch_threshold.cpp # Threshold protocol implementation
```

### Patent

See `/Users/z/work/lux/patents/fhe/PAT-FHE-016-batched-threshold-fhe-protocol.md`.

## GPU Patent Implementations (10 Optimizations)

All implemented in `src/core/lib/math/hal/mlx/`:

| ID | Optimization | Files |
|----|-------------|-------|
| A1 | Unified-memory NTT streaming | `ntt_unified_memory.metal`, `unified_stream.h` |
| A2 | Four-step NTT Apple Metal | `four_step_ntt.metal`, `ntt_four_step.metal`, `four_step_ntt.h` |
| A3 | Fused external product kernel | `fused_external_product.metal`, `external_product_fused.metal`, `fused_external_product.h` |
| A4 | Twiddle hotset caching | `twiddle_cache.metal`, `twiddle_cache.h`, `twiddle_cache.cpp` |
| B5 | Threshold batch GPU layout | `batch_threshold.h` |
| B6 | Speculative BSK prefetch | `bsk_prefetch.metal` |
| C7 | EVM-FHE lazy carry model | `euint256.h` |
| C8 | Encrypted comparison Solidity | `euint256.h` (comparison ops) |
| C9 | Verifiable FHE witness | `threshold/` module |
| D10 | GPU scheme switching | `scheme_switch.metal` |

### Library Naming

Libraries renamed from `OPENFHE*` to cleaner names:
- `libFHEcore.dylib` - Core FHE operations
- `libFHEbin.dylib` - Binary/Boolean FHE (TFHE-style)
- `libFHEpke.dylib` - Public key encryption schemes (CKKS, BGV, BFV)

### Licensing

**Open Source (BSD-3-Clause):**
- Apple Silicon / Metal / MLX acceleration
- All code in this repository

**Enterprise (Contact licensing@lux.network):**
- NVIDIA CUDA acceleration
- Multi-GPU support (H100/H200/HGX)
- Datacenter deployment
- See LP-0050 for details

## References

- [OpenFHE Documentation](https://openfhe-development.readthedocs.io/)
- [MLX Framework](https://ml-explore.github.io/mlx/)
- [Metal Shading Language](https://developer.apple.com/metal/)
- [LMKCDEY Paper](https://eprint.iacr.org/2022/198)
- [Lux Lattice Library](https://github.com/luxfi/lattice)

---

*Last Updated: 2025-12-30 - Updated architecture to use luxcpp/gpu foundation*
