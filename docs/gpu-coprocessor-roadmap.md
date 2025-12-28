# Lux FHE GPU Coprocessor Roadmap

## Goal
Build an fhEVM GPU coprocessor using OpenFHE BinFHE (GINX), with encrypted bool + uint* semantics, optimized for MLX/CUDA acceleration.

## Architecture Layers

### 1. Semantics Layer - Radix Integer Runtime
Build shortint/radix integer APIs on top of OpenFHE BinFHE's EvalFunc + CMUX.

```
euint256 = limbs[] where each limb is an LWE ciphertext in Z_p
```

**Required Primitives:**
- `add_limb_with_carry(a, b, c) → (sum, carry)` via LUT
- `sub_limb_with_borrow(a, b, br) → (diff, borrow)` via LUT  
- `mul_limb(a, b)` (expensive, optional for MVP)
- `eq/lt/le` via LUT + borrow propagation + CMUX
- `SELECT/CMUX` for encrypted control flow

**Files to create:**
- `src/binfhe/shortint/` - Shortint module
- `src/binfhe/radix/` - Radix integer composition
- `go/radix/` - Go bindings for radix types

### 2. Execution Layer - Batch APIs
OpenFHE has scalar Bootstrap/EvalFunc. Need batch APIs to saturate GPU.

**Required APIs:**
```cpp
// Batch bootstrapping
void BootstrapBatch(
    ParamsId paramsId,
    const std::vector<LWECiphertext>& ct_in,
    std::vector<LWECiphertext>& ct_out,
    uint32_t flags = 0
);

// Batch function evaluation
void EvalFuncBatch(
    ParamsId paramsId,
    const std::vector<LWECiphertext>& ct_in,
    const LUT& lut,
    std::vector<LWECiphertext>& ct_out
);

// Batch key switching
void KeySwitchBatch(
    ParamsId paramsId,
    const std::vector<LWECiphertext>& ct_in,
    std::vector<LWECiphertext>& ct_out
);

// Batch modulus switching
void ModSwitchBatch(
    ParamsId paramsId,
    const std::vector<LWECiphertext>& ct_in,
    std::vector<LWECiphertext>& ct_out
);
```

**Files to modify:**
- `src/binfhe/lib/binfhecontext.cpp` - Add batch methods
- `src/binfhe/include/binfhecontext.h` - Declare batch APIs

### 3. Engine Layer - GPU Backend

#### 3.1 Packed Device Formats
Canonical binary layouts for zero-copy GPU transfer:

```cpp
// Packed LWE ciphertext
struct PackedLWECt {
    uint32_t version;
    uint32_t n;          // LWE dimension
    uint32_t log_q;      // Modulus bits
    int64_t* data;       // [a_0, ..., a_{n-1}, b] packed
};

// Packed bootstrapping key
struct PackedBTKey {
    uint32_t version;
    uint32_t n;          // Input LWE dimension
    uint32_t N;          // Ring dimension
    uint32_t k;          // RLWE dimension  
    uint32_t base_g;     // Gadget base
    uint32_t num_levels; // Decomposition levels
    int64_t* data;       // RGSW samples packed row-major
};
```

**Required APIs:**
```cpp
std::vector<uint8_t> ExportBTKeyPacked(ParamsId paramsId);
void ImportBTKeyPacked(ParamsId paramsId, const std::vector<uint8_t>& data);
std::vector<uint8_t> ExportSwitchKeyPacked(ParamsId paramsId);
void ImportSwitchKeyPacked(ParamsId paramsId, const std::vector<uint8_t>& data);
```

#### 3.2 Backend Abstraction
Split GINX blind rotation into pluggable backend:

```cpp
class BinFHEBackend {
public:
    virtual ~BinFHEBackend() = default;
    
    // Core blind rotation
    virtual void BlindRotate(
        RingGSWACCKey& bk,
        RLWE& acc,
        const std::vector<NativeInteger>& a,
        const NativeInteger& mod
    ) = 0;
    
    // External product
    virtual void ExternalProduct(
        RingGSWCiphertext& ct,
        RLWE& acc,
        NativeInteger scale
    ) = 0;
    
    // Sample extraction
    virtual LWECiphertext SampleExtract(
        const RLWE& acc,
        uint32_t index
    ) = 0;
};

// Implementations
class BackendCPU : public BinFHEBackend { ... };  // Existing OpenFHE code
class BackendMLX : public BinFHEBackend { ... };  // MLX (CUDA/Metal) kernels
```

**Files to create:**
- `src/binfhe/lib/backend/backend.h` - Interface
- `src/binfhe/lib/backend/backend_cpu.cpp` - CPU impl (refactor existing)
- `src/binfhe/lib/backend/backend_mlx.cpp` - MLX/CUDA impl

#### 3.3 MLX/CUDA Kernels
Deterministic integer kernels only (no floats - need bit-identical for consensus):

```cpp
// Gadget decomposition kernel
__global__ void gadget_decompose(
    const int64_t* input,    // [batch, n]
    int64_t* output,         // [batch, n, levels]
    uint32_t n,
    uint32_t base_g,
    uint32_t num_levels
);

// External product kernel (NTT-domain)
__global__ void external_product_ntt(
    const int64_t* acc,      // [batch, N]
    const int64_t* bk,       // [n, k, levels, N]
    const int64_t* decomp,   // [batch, n, levels]
    int64_t* result,         // [batch, N]
    uint32_t N,
    uint32_t n,
    uint32_t k,
    uint32_t levels
);

// Blind rotation kernel
__global__ void blind_rotate(
    const int64_t* acc_in,
    const int64_t* bk,
    const int32_t* lwe_a,    // [batch, n]
    int64_t* acc_out,
    uint32_t batch,
    uint32_t n,
    uint32_t N
);
```

### 4. Parameter Profiles

Opinionated, tested parameter sets:

```cpp
enum FHEVMProfile {
    SHORTINT_K4_STD128_FAST,   // 4-bit limbs, 128-bit security, optimized
    SHORTINT_K8_STD128_FAST,   // 8-bit limbs, 128-bit security, optimized
    RADIX_UINT256_PROFILE_A,   // 256-bit integers, balanced
    RADIX_UINT256_PROFILE_B,   // 256-bit integers, throughput-optimized
};

struct ProfileParams {
    uint32_t limb_bits;        // k = log2(p)
    uint32_t num_limbs;        // For radix: 256/k
    uint32_t bootstrap_freq;   // Bootstrap every N ops
    uint32_t carry_strategy;   // 0=eager, 1=lazy
    // ... standard OpenFHE params
};
```

### 5. Multi-User & Threshold

**Not in OpenFHE BinFHE**, need to implement:

- Handle ACLs (who can request decrypt/reencrypt)
- Re-encryption protocol using keyswitching
- Threshold key management (separate from BinFHE)

This lives in:
- `go/threshold/` - Threshold FHE integration
- Integration with `thresholdvm` in node

## Implementation Priority

### Phase 1: Foundation (Current)
- [x] Basic TFHE Go bindings
- [x] Solidity FHE.sol contracts
- [x] Coreth precompile scaffolding
- [ ] Packed key formats
- [ ] Batch APIs (BootstrapBatch, EvalFuncBatch)

### Phase 2: GPU Backend
- [ ] Backend abstraction in GINX
- [ ] MLX/CUDA kernel stubs
- [ ] Device key cache
- [ ] Benchmark harness

### Phase 3: Radix Integers
- [ ] Shortint module (LUT-based arithmetic)
- [ ] Radix composition
- [ ] Carry propagation strategies
- [ ] Go bindings for euint8...euint256

### Phase 4: Production
- [ ] Full MLX/CUDA kernel implementation
- [ ] Threshold integration
- [ ] Gas cost calibration
- [ ] Lux fhEVM contract testing

## OpenFHE Source Map

Key files to modify in `src/binfhe/`:

| File | Purpose | Changes Needed |
|------|---------|----------------|
| `lib/binfhecontext.cpp` | Main context | Add batch APIs, backend selection |
| `lib/rgsw-acc-cggi.cpp` | GINX accumulator | Extract backend interface |
| `lib/lwe-pke.cpp` | LWE primitives | Add packed formats |
| `include/binfhecontext.h` | Headers | Declare new APIs |

## References

- [OpenFHE BinFHE Docs](https://openfhe-development.readthedocs.io/en/latest/sphinx_rsts/modules/binfhe.html)
- [TFHE Paper](https://eprint.iacr.org/2018/421)
- [Fast Blind Rotation](https://eprint.iacr.org/2023/958)
- [Lux FHE Library](https://github.com/luxfi/fhe)
