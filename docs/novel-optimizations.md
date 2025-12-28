# Lux FHE Novel Optimizations

## Patent Strategy

This document describes novel optimizations in Lux FHE that are unique to our blockchain-focused implementation and could be considered for patent protection.

### Known Patents to Avoid

We explicitly **avoid** the following patented techniques:

| Patent | Description | Our Approach |
|--------|-------------|--------------|
| EP4195578 | Seed + Fourier ciphertext storage for PBS | Use standard RLWE representation |
| EP4488821 | Shift-left PBS shift-right error reduction | Use classical bootstrapping refresh |
| WO2023067928 | Integer-wise TFHE arithmetic circuits | Novel limb composition (see below) |
| WO2023074133 | TFHE integer operations | Alternative carry propagation |

### Safe Prior Art We Build On

- **Chillotti et al. ASIACRYPT 2017** - Original TFHE programmable bootstrapping
- **FHEW (Ducas-Micciancio 2015)** - Binary gate bootstrapping
- **Generic LWE/RLWE** - Standard lattice cryptography

---

## Novel Optimization 1: Consensus-Integrated Threshold FHE

**Problem**: Existing threshold FHE requires separate communication rounds for distributed decryption, adding latency to blockchain finality.

**Innovation**: Integrate threshold FHE decryption into the Lux Snow++ consensus protocol:

```
┌─────────────────────────────────────────────────────────────┐
│                    Lux Consensus Round                       │
├─────────────────────────────────────────────────────────────┤
│  1. Transaction includes FHE ciphertext requiring decrypt   │
│  2. During consensus sampling, validators:                   │
│     a. Vote on block validity                               │
│     b. Include partial decryption share in vote             │
│  3. Block proposer aggregates:                              │
│     a. Consensus votes → finality                           │
│     b. Decryption shares → plaintext                        │
│  4. Single-round threshold decrypt + finality               │
└─────────────────────────────────────────────────────────────┘
```

**Claims**:
1. Method for combining threshold FHE partial decryption with blockchain consensus voting
2. Single-round protocol achieving both block finality and ciphertext decryption
3. Validator selection for decryption based on stake-weighted sampling

---

## Novel Optimization 2: Transaction-Batch Amortized Bootstrapping

**Problem**: Each FHE operation requires expensive bootstrapping (~13ms). Blockchain transactions arrive in batches but are processed independently.

**Innovation**: Batch bootstrap keys across transactions in a block:

```
Traditional:
  Tx1: [op1 → bootstrap → op2 → bootstrap]  Total: N×bootstrap
  Tx2: [op1 → bootstrap → op2 → bootstrap]
  Tx3: [op1 → bootstrap → op2 → bootstrap]

Lux Batched:
  Block: [Tx1.op1, Tx2.op1, Tx3.op1] → BATCH_BOOTSTRAP → [Tx1.op2, Tx2.op2, Tx3.op2]
  Total: ceil(N/batch_size)×bootstrap
```

**Key insight**: EVM execution is deterministic - we can analyze the FHE operation DAG across all transactions before execution and schedule bootstraps optimally.

**Claims**:
1. Method for analyzing FHE operation dependencies across multiple blockchain transactions
2. Cross-transaction bootstrap batching for GPU throughput
3. DAG-based scheduling minimizing total bootstrap operations per block

---

## Novel Optimization 3: Lazy Carry Propagation with Deterministic Noise Tracking

**Problem**: Radix integer arithmetic requires carry propagation via bootstrapping. Existing implementations bootstrap after every operation.

**Innovation**: Track noise accumulation deterministically and defer carries:

```cpp
// Traditional: bootstrap after each add
result = add(a, b);  // bootstrap
result = add(result, c);  // bootstrap
result = add(result, d);  // bootstrap
// 3 bootstraps

// Lux Lazy Carry:
result = add_lazy(a, b);  // accumulate noise
result = add_lazy(result, c);  // accumulate noise
result = add_lazy(result, d);  // accumulate noise
result = propagate_if_needed(result);  // 1 bootstrap (if noise exceeds threshold)
// 0-1 bootstraps depending on noise budget
```

**Key insight**: The 2-bit carry buffer in our limb representation allows 2-3 additions before overflow. We track noise deterministically (not probabilistically) based on operation count.

**Claims**:
1. Deterministic noise budget tracking for FHE radix integers
2. Lazy carry propagation with configurable bootstrap threshold
3. Method for deferring FHE bootstrapping based on operation history

---

## Novel Optimization 4: Subnet-Specific FHE Parameters

**Problem**: Different blockchain applications have different security/performance tradeoffs. Single parameter set is suboptimal.

**Innovation**: Per-subnet configurable FHE parameters:

```
Subnet A (High-frequency DeFi):
  - Security: 128-bit
  - Message bits: 4 per limb
  - Bootstrap: ~8ms (faster, lower precision)

Subnet B (Confidential Voting):
  - Security: 256-bit
  - Message bits: 2 per limb
  - Bootstrap: ~20ms (slower, higher security)

Subnet C (Privacy-Preserving ML):
  - Security: 128-bit
  - CKKS mode for approximate arithmetic
  - No bootstrapping (leveled)
```

**Claims**:
1. Blockchain subnet architecture with per-subnet FHE parameter selection
2. Cross-subnet encrypted data migration with parameter conversion
3. Dynamic security level adjustment based on subnet policy

---

## Novel Optimization 5: Precompile Gas Metering for FHE Operations

**Problem**: FHE operations have highly variable cost. Flat gas pricing leads to DoS vectors or underpriced operations.

**Innovation**: Dynamic gas pricing based on operation complexity:

```solidity
// Gas cost = base + (type_bits × bit_cost) + (op_complexity × complexity_cost)

function estimateGas(FheOp op, FheType type) returns (uint256) {
    uint256 base = 10000;
    uint256 bits = typeBits(type);
    
    if (op == FheOp.ADD || op == FheOp.SUB) {
        // Linear in bits
        return base + bits * 500;
    } else if (op == FheOp.MUL) {
        // Quadratic in bits (schoolbook)
        return base + bits * bits * 50;
    } else if (op == FheOp.DIV) {
        // Cubic in bits
        return base + bits * bits * bits * 5;
    }
    // ...
}
```

**Claims**:
1. Method for computing EVM gas costs for FHE operations based on encrypted type width
2. Operation-specific gas formulae reflecting cryptographic complexity
3. Dynamic gas adjustment based on current FHE coprocessor load

---

## Novel Optimization 6: Encrypted Index Private Information Retrieval

**Problem**: Smart contracts accessing encrypted arrays leak access patterns.

**Innovation**: FHE-native PIR using programmable bootstrapping:

```cpp
// Traditional: access pattern leaked
encrypted_value = array[encrypted_index];  // Server sees which index

// Lux PIR:
// 1. Client encrypts index
// 2. For each position, compute: select(eq(i, encrypted_index), array[i], zero)
// 3. Sum all positions → encrypted result at encrypted index
// Server learns nothing about access pattern

encrypted_value = fhe_pir(array, encrypted_index);
```

**Key insight**: The select (CMUX) operation can be batched efficiently for PIR.

**Claims**:
1. Method for private information retrieval using FHE select operations
2. Batched CMUX evaluation for encrypted array access
3. Smart contract pattern for oblivious array indexing

---

## Novel Optimization 7: Validator Keyshare Rotation Without Downtime

**Problem**: Threshold FHE requires key resharing when validator set changes. Naive approach requires downtime.

**Innovation**: Proactive secret sharing with encrypted keyshare migration:

```
Epoch N validators: {V1, V2, V3}  holding shares {s1, s2, s3}
Epoch N+1 validators: {V2, V3, V4}

1. V1 (leaving) encrypts their share to V4 (joining) using V4's public key
2. V2, V3 participate in MPC to re-randomize shares
3. V4 decrypts their new share
4. New threshold set {V2, V3, V4} can decrypt
5. Old share s1 is information-theoretically destroyed

No downtime: decryption works throughout transition
```

**Claims**:
1. Method for threshold FHE keyshare rotation during validator set changes
2. Encrypted keyshare migration between validators
3. Zero-downtime threshold key refresh protocol

---

## Implementation Notes

### Patent-Safe Design Principles

1. **Use pre-2020 academic techniques** - Chillotti TFHE, FHEW bootstrapping
2. **Novel composition, not novel primitives** - Our innovation is in blockchain integration
3. **Avoid seed+Fourier storage** - Use standard polynomial representation
4. **Avoid patented error reduction** - Use classical noise refresh

### Files Implementing These Optimizations

| Optimization | Primary Files |
|--------------|---------------|
| Consensus-Integrated TFHE | `threshold/consensus.cpp` (planned) |
| Batch Bootstrapping | `batch/binfhe-batch.h`, `batch/batch.cpp` |
| Lazy Carry | `radix/radix.cpp` - `PropagateCarries()` |
| Subnet Parameters | `fhevm/fhevm.cpp` - `FheContext` constructor |
| Gas Metering | `fhevm/fhevm.cpp` - `EstimateGas()` |
| Encrypted PIR | `fhevm/pir.cpp` (planned) |
| Keyshare Rotation | `threshold/rotation.cpp` (planned) |

---

## Prior Art Analysis

Before filing patents, verify these are not covered by:

- [ ] TFHE paper (Chillotti et al.)
- [ ] FHEW paper (Ducas-Micciancio)
- [ ] Lattigo library (EPFL)
- [ ] HElib (IBM)
- [ ] SEAL (Microsoft)
- [ ] FHE-related patents (EP4195578, WO2023067928, EP4488821, etc.)
- [ ] Intel/AMD FHE acceleration patents

---

## Novel Optimization 8: Deterministic FHE Random Number Generation

**Problem**: FHE operations require random sampling (noise, blinding), but blockchain requires deterministic execution across all nodes.

**Innovation**: SHA256-based deterministic PRNG seeded from blockchain state:

```go
type FheRNG struct {
    state   [32]byte  // SHA256 state
    counter uint64    // Monotonic counter
}

func (rng *FheRNG) advance() [32]byte {
    data := append(rng.state[:], counter...)
    rng.state = sha256.Sum256(data)
    rng.counter++
    return rng.state
}

// Seed from blockchain: sha256(blockHash || txHash || opIndex)
```

**Claims**:
1. Method for generating deterministic encrypted random values using cryptographic hash-based state machine
2. System for blockchain-compatible FHE randomness where all validators produce identical ciphertext outputs
3. Computer-implemented method for seeding FHE operations from blockchain state

**Files**: `luxfi/tfhe/random.go`

---

## Novel Optimization 9: Pure Go TFHE Without CGO

**Problem**: C/C++ FHE libraries require CGO for Go integration, limiting cloud deployment and increasing complexity.

**Innovation**: Complete TFHE implementation in pure Go with no foreign function interfaces:

```go
// Pure Go blind rotation
func (eval *Evaluator) bootstrap(ct *Ciphertext, testPoly *ring.Poly) (*Ciphertext, error) {
    testPolyMap := map[int]*ring.Poly{0: testPoly}
    results, err := eval.eval.Evaluate(ct.Ciphertext, testPolyMap, eval.bsk.BRK)
    // ... pure Go implementation using luxfi/lattice
}
```

**Benefits**:
- No C compiler required for deployment
- Cross-platform binary compilation (GOOS, GOARCH)
- Easier cloud/serverless deployment
- Memory safety guarantees from Go runtime

**Claims**:
1. Pure Go implementation of TFHE programmable bootstrapping without foreign function interfaces
2. Method for deploying FHE operations in cloud environments without native code dependencies
3. Cross-platform FHE execution system using managed runtime languages

**Files**: `luxfi/tfhe/*.go`, `luxfi/lattice/schemes/tfhe/*.go`

---

## Novel Optimization 10: Batch DAG Execution with Async Futures

**Problem**: Individual FHE operations have high overhead. GPU utilization is low with sequential execution.

**Innovation**: DAG-based scheduling with async futures and multi-output evaluation:

```cpp
class BatchDAG {
    size_t AddBootstrap(size_t input_id);
    size_t AddEvalFunc(size_t input_id, const std::vector<NativeInteger>& lut);
    size_t AddBinGate(BINGATE gate, size_t input1_id, size_t input2_id);
    BatchResult Execute(uint32_t flags = BATCH_DEFAULT);
};

// Multi-output: produce (sum, carry) from single add_with_carry
BatchResult EvalFuncMultiOutputBatch(
    const std::vector<LWECiphertext>& ct_in,
    const std::vector<std::vector<NativeInteger>>& luts,  // Multiple LUTs
    std::vector<LWECiphertext>& ct_out
);
```

**Claims**:
1. DAG-based scheduler for FHE operations enabling optimal GPU batching
2. Multi-output batch function evaluation for FHE radix arithmetic
3. Async batch processing system for FHE with future-based result retrieval

**Files**: `fhe/src/binfhe/include/batch/binfhe-batch.h`

---

## Patent Strategy Summary

See **LP-8101** (`lps/LPs/lp-8101-fhe-patent-strategy.md`) for complete patent strategy including:
- 10 patentable innovations
- Priority ranking
- Filing jurisdictions
- Prosecution timeline
- Prior art analysis

---

*Document prepared for Lux Industries Inc patent review. Not legal advice.*
