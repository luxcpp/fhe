# GPU TFHE Architecture Design for Apple Metal

## Executive Summary

This document specifies the optimal GPU kernel architecture for TFHE blind rotation and key switching operations targeting Apple Silicon (M1/M2/M3) via the MLX framework. The design prioritizes:

1. **Fused kernels** to minimize memory transfers
2. **Coalesced memory access** patterns for maximum bandwidth
3. **Massive parallelism** exploiting batch, component, and coefficient dimensions
4. **Zero CPU roundtrips** during the critical path

---

## 1. System Architecture Overview

```
                    +------------------------------------------+
                    |           GPU TFHE Engine                |
                    |                                          |
   LWE Input        |  +----------------+  +----------------+  |   LWE Output
   [B, n+1]    ---->|  | Blind Rotation |->| Key Switching  |  |---> [B, n+1]
                    |  +----------------+  +----------------+  |
                    |         |                    |           |
                    |    +----v----+          +----v----+      |
                    |    |  NTT    |          |  INTT   |      |
                    |    +---------+          +---------+      |
                    |         |                    ^           |
                    |    +----v--------------------+----+      |
                    |    |    External Product Engine   |      |
                    |    +------------------------------+      |
                    +------------------------------------------+
                                     |
                              +------v------+
                              | Metal Shader|
                              |   Library   |
                              +-------------+
```

---

## 2. Data Flow: Full Bootstrap Pipeline

### 2.1 High-Level Flow

```
LWE Ciphertext (q = 2^15)
        |
        v
[Modulus Switch: q -> 2N]
        |
        v
[Initialize Accumulator: acc = X^{-b} * TestPoly]
        |
        +---> [NTT Transform acc to NTT domain]
        |
        v
+-----------------------------------+
| For i = 0 to n-1:                 |
|   if a[i] != 0:                   |
|     rotated = X^{a[i]} * acc      | <-- Negacyclic rotation (NTT domain)
|     diff = rotated - acc          |
|     acc += ExtProd(diff, BK[i])   | <-- Fused external product
+-----------------------------------+
        |
        v
[Extract RLWE constant term]
        |
        v
[Key Switch: RLWE(N) -> LWE(n)]
        |
        v
LWE Ciphertext (output)
```

### 2.2 Detailed Data Flow Diagram

```
                          BATCH INPUT: [B, n+1]
                                   |
                    +--------------+--------------+
                    |                             |
               a[B, n]                        b[B]
                    |                             |
                    v                             v
            +---------------+             +---------------+
            | Mod Switch    |             | Mod Switch    |
            | a' = a * 2N/q |             | b' = b * 2N/q |
            +---------------+             +---------------+
                    |                             |
                    |              +--------------+
                    |              |
                    v              v
            +-------------------------+
            | Init Accumulator        |
            | acc1[B,N] = Rotate(     |
            |   TestPoly, -b'[B])     |
            | acc0[B,N] = 0           |
            +-------------------------+
                         |
                         v
            +-------------------------+
            | Forward NTT (Fused)     |
            | acc0_ntt, acc1_ntt      |
            +-------------------------+
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
    [BK[0]]         [BK[1]]   ...  [BK[n-1]]
          |              |              |
          v              v              v
    +------------+  +------------+  +------------+
    | CMux Gate  |  | CMux Gate  |  | CMux Gate  |
    | a'[i]=a[0] |  | a'[i]=a[1] |  | a'[i]=... |
    +------------+  +------------+  +------------+
          |              |              |
          +----> ... --->+---> ... ---->+
                                        |
                                        v
                         +-------------------------+
                         | Key Switch Decompose    |
                         | digits[B,N,L_ks]        |
                         +-------------------------+
                                        |
                                        v
                         +-------------------------+
                         | Key Switch Accumulate   |
                         | output[B, n+1]          |
                         +-------------------------+
                                        |
                                        v
                              OUTPUT: [B, n+1]
```

---

## 3. Memory Layout Specifications

### 3.1 Structure of Arrays (SoA) for Coalesced Access

The key to achieving high GPU bandwidth is ensuring that adjacent threads access adjacent memory locations. We use SoA layout throughout.

```cpp
// LWE Ciphertext Batch: [B, n+1]
// Layout: Row-major, batch dimension outermost
struct LWEBatchLayout {
    // Memory: [B * n] contiguous for masks
    //         [B] contiguous for bodies
    
    // Access pattern for thread (batch_idx, coeff_idx):
    //   a[batch_idx * n + coeff_idx]  <-- coalesced across coeff_idx
    
    uint64_t* a;  // [B, n] - mask coefficients
    uint64_t* b;  // [B]    - body values
};

// RLWE Ciphertext Batch: [B, 2, N]  
// Layout: [batch, component, coefficient]
struct RLWEBatchLayout {
    // Access: data[batch * 2 * N + comp * N + coeff]
    // Coalesced access when iterating over coefficients
    
    uint64_t* data;  // [B * 2 * N]
    
    // Alternative split layout:
    uint64_t* c0;    // [B, N] - first polynomial
    uint64_t* c1;    // [B, N] - second polynomial
};

// Bootstrap Key: [n, 2, L, 2, N]
// Layout optimized for sequential access during blind rotation
struct BootstrapKeyLayout {
    // For BK[i], we need all L levels of both RGSW rows
    // Layout: [n, 2, L, 2, N] = [n][2][L][2][N]
    
    // Access pattern in external product:
    //   For each LWE index i (0..n-1):
    //     For each decomposition level l (0..L-1):
    //       Access BK[i, :, l, :, :] = 2 * 2 * N values
    
    uint64_t* data;  // Total: n * 2 * L * 2 * N uint64_t values
    
    // Index calculation:
    // BK[i][row][l][col][coeff] = 
    //   data[i * (2*L*2*N) + row * (L*2*N) + l * (2*N) + col * N + coeff]
};

// Key Switch Key: [N, L_ks, n+1]
struct KeySwitchKeyLayout {
    // For each RLWE coefficient j (0..N-1):
    //   For each decomposition level l (0..L_ks-1):
    //     LWE ciphertext of dimension n
    
    uint64_t* data;  // Total: N * L_ks * (n+1) values
};
```

### 3.2 Memory Alignment Requirements

```cpp
// Alignment for optimal Metal buffer access
constexpr size_t METAL_ALIGNMENT = 256;  // 256-byte alignment

// Page alignment for large allocations
constexpr size_t PAGE_SIZE = 16384;  // 16KB pages on Apple Silicon

// Ensure all major buffers are page-aligned
template<typename T>
T* allocateAligned(size_t count) {
    size_t bytes = count * sizeof(T);
    bytes = (bytes + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
    void* ptr = nullptr;
    posix_memalign(&ptr, PAGE_SIZE, bytes);
    return static_cast<T*>(ptr);
}
```

### 3.3 Threadgroup Shared Memory Layout

```cpp
// Shared memory organization for NTT kernel (per threadgroup)
// Threadgroup size: 256 threads for N=1024
struct NTTSharedMemory {
    // Double-buffered polynomial data for ping-pong
    uint64_t poly_a[1024];  // 8KB
    uint64_t poly_b[1024];  // 8KB
    
    // Twiddle factor cache (frequently accessed)
    uint64_t twiddles[512]; // 4KB - half of twiddles fit
    
    // Total: 20KB < 32KB threadgroup memory limit
};

// Shared memory for external product (per threadgroup)
struct ExtProdSharedMemory {
    // Decomposed digits for current RLWE component
    uint64_t digits[4][1024];  // L=4 levels, N=1024 coeffs = 32KB
    
    // Note: At L=4 and N=1024, we're at the limit
    // May need to process in chunks for larger L
};
```

---

## 4. Kernel Architecture

### 4.1 Fused Blind Rotation Kernel

The blind rotation is the most compute-intensive operation. We fuse multiple steps into a single kernel launch sequence.

```
+------------------------------------------------------------------+
|                    FUSED BLIND ROTATION KERNEL                    |
+------------------------------------------------------------------+
|                                                                  |
|  Input:                                                          |
|    - lwe[B, n+1]: LWE ciphertexts (modulus-switched)            |
|    - bsk[n, 2, L, 2, N]: Bootstrap keys (NTT domain)            |
|    - test_poly[N]: Test polynomial (coefficient domain)          |
|                                                                  |
|  Output:                                                         |
|    - acc[B, 2, N]: RLWE accumulators (NTT domain)               |
|                                                                  |
|  Kernel Stages:                                                  |
|                                                                  |
|  Stage 0: Initialize Accumulators                                |
|  +------------------------------------------------------------+  |
|  | Grid: [N/256, B, 1]  Threads: [256, 1, 1]                  |  |
|  | For each batch b, coefficient c:                           |  |
|  |   rotation = -lwe[b, n] mod 2N                             |  |
|  |   src = (c + rotation) mod 2N                              |  |
|  |   sign = (src >= N) ? -1 : 1                               |  |
|  |   acc[b, 0, c] = 0                                         |  |
|  |   acc[b, 1, c] = sign * test_poly[src mod N]               |  |
|  +------------------------------------------------------------+  |
|                              |                                   |
|                              v                                   |
|  Stage 1: Forward NTT on Accumulators                            |
|  +------------------------------------------------------------+  |
|  | Grid: [N/256, B, 2]  Threads: [256, 1, 1]                  |  |
|  | 10 substages (log2(1024) = 10)                             |  |
|  | Each substage: butterfly operations with barrier           |  |
|  +------------------------------------------------------------+  |
|                              |                                   |
|                              v                                   |
|  Stage 2: CMux Loop (n iterations, fused)                        |
|  +------------------------------------------------------------+  |
|  | For i = 0 to n-1:                                          |  |
|  |   Grid: [N/256, B, 2]                                      |  |
|  |                                                            |  |
|  |   Substage 2a: Compute rotated accumulator                 |  |
|  |   +--------------------------------------------------------+  |
|  |   | rotation[b] = lwe[b, i] mod 2N                         |  |
|  |   | rotated[b,c,coeff] = NegacyclicRotate(acc, rotation)   |  |
|  |   | diff[b,c,coeff] = rotated - acc                        |  |
|  |   +--------------------------------------------------------+  |
|  |                                                            |  |
|  |   Substage 2b: Decompose diff into L digits                |  |
|  |   +--------------------------------------------------------+  |
|  |   | For level l in 0..L-1:                                 |  |
|  |   |   digits[b,c,l,coeff] = (diff >> (l*baseLog)) & mask   |  |
|  |   +--------------------------------------------------------+  |
|  |                                                            |  |
|  |   Substage 2c: External product accumulation               |  |
|  |   +--------------------------------------------------------+  |
|  |   | For out_c in 0..1:                                     |  |
|  |   |   sum = 0                                              |  |
|  |   |   For in_c in 0..1:                                    |  |
|  |   |     For l in 0..L-1:                                   |  |
|  |   |       sum += digits[in_c,l] * bsk[i,in_c,l,out_c]      |  |
|  |   |   acc[b,out_c,coeff] += sum                            |  |
|  |   +--------------------------------------------------------+  |
|  +------------------------------------------------------------+  |
|                              |                                   |
+------------------------------v-----------------------------------+
                        acc[B, 2, N]
```

### 4.2 Optimized NTT Kernel

```
+------------------------------------------------------------------+
|                    FUSED NTT KERNEL (Forward)                     |
+------------------------------------------------------------------+
|                                                                  |
|  Configuration:                                                  |
|    N = 1024 (ring dimension)                                     |
|    LOG_N = 10 (stages)                                           |
|    THREADS_PER_BLOCK = 256                                       |
|    BLOCKS_PER_POLY = N / (2 * THREADS_PER_BLOCK) = 2             |
|                                                                  |
|  Memory:                                                         |
|    Input/Output: device memory [B, N]                            |
|    Twiddles: constant memory [N]                                 |
|    Scratch: threadgroup memory [1024]                            |
|                                                                  |
|  Algorithm: Cooley-Tukey with shared memory optimization         |
|                                                                  |
|  +------------------------------------------------------------+  |
|  | Phase 1: Global memory stages (stages 0-5)                 |  |
|  |   - Large butterflies, stride > threadgroup size           |  |
|  |   - Each thread handles one butterfly                      |  |
|  |   - 6 kernel launches with global sync                     |  |
|  +------------------------------------------------------------+  |
|                              |                                   |
|  +------------------------------------------------------------+  |
|  | Phase 2: Shared memory stages (stages 6-9)                 |  |
|  |   - Load 1024 elements to threadgroup memory               |  |
|  |   - Process 4 stages within threadgroup                    |  |
|  |   - Single kernel, threadgroup barriers                    |  |
|  |   - Write back to global memory                            |  |
|  +------------------------------------------------------------+  |
|                                                                  |
|  Optimization: Batch multiple polynomials per kernel launch     |
|    Grid: [N/512, B, 1]                                          |
|    Each threadgroup processes 512 coefficients                   |
|                                                                  |
+------------------------------------------------------------------+
```

### 4.3 External Product Kernel

```
+------------------------------------------------------------------+
|                    EXTERNAL PRODUCT KERNEL                        |
+------------------------------------------------------------------+
|                                                                  |
|  Operation: RLWE x RGSW -> RLWE  (in NTT domain)                 |
|                                                                  |
|  Input:                                                          |
|    rlwe[B, 2, N]: RLWE ciphertexts (NTT domain)                 |
|    rgsw[B, 2, L, 2, N]: RGSW ciphertexts (NTT domain)           |
|                                                                  |
|  Output:                                                         |
|    result[B, 2, N]: RLWE result (NTT domain)                    |
|                                                                  |
|  Thread Organization:                                            |
|    Grid: [N/256, 2, B]                                          |
|    Threads: [256, 1, 1]                                         |
|    Total threads: (N/256) * 2 * B * 256 = 2*N*B                 |
|                                                                  |
|  Algorithm per thread (batch b, output component out_c, coeff):  |
|                                                                  |
|  +------------------------------------------------------------+  |
|  | // Step 1: Decompose RLWE coefficients into digits         |  |
|  | for (in_c = 0; in_c < 2; in_c++) {                         |  |
|  |     uint64_t val = rlwe[b, in_c, coeff];                   |  |
|  |     for (l = 0; l < L; l++) {                              |  |
|  |         digits[in_c][l] = (val >> (l * BASE_LOG)) & MASK;  |  |
|  |     }                                                      |  |
|  | }                                                          |  |
|  +------------------------------------------------------------+  |
|  | // Step 2: Accumulate products                             |  |
|  | uint64_t sum = 0;                                          |  |
|  | for (in_c = 0; in_c < 2; in_c++) {                         |  |
|  |     for (l = 0; l < L; l++) {                              |  |
|  |         uint64_t g = rgsw[b, in_c, l, out_c, coeff];       |  |
|  |         sum = modadd(sum, modmul(digits[in_c][l], g));     |  |
|  |     }                                                      |  |
|  | }                                                          |  |
|  | result[b, out_c, coeff] = sum;                             |  |
|  +------------------------------------------------------------+  |
|                                                                  |
|  Memory Access Pattern:                                          |
|    - rlwe: stride-N access (coalesced within component)          |
|    - rgsw: complex indexing, prefetch into registers             |
|    - result: stride-N write (coalesced)                          |
|                                                                  |
+------------------------------------------------------------------+
```

### 4.4 Key Switching Kernel

```
+------------------------------------------------------------------+
|                    KEY SWITCHING KERNEL                           |
+------------------------------------------------------------------+
|                                                                  |
|  Operation: RLWE(N) -> LWE(n) via key switching                  |
|                                                                  |
|  Input:                                                          |
|    rlwe[B, 2, N]: RLWE ciphertexts (after blind rotation)       |
|    ksk[N, L_ks, n+1]: Key switching keys                        |
|                                                                  |
|  Output:                                                         |
|    lwe[B, n+1]: LWE ciphertexts                                 |
|                                                                  |
|  Algorithm:                                                      |
|                                                                  |
|  +------------------------------------------------------------+  |
|  | Stage 1: Extract and INTT the constant term                |  |
|  |   (Or extract from coefficient domain if acc not in NTT)   |  |
|  |                                                            |  |
|  |   For RLWE (c0, c1), the decryption is:                    |  |
|  |     m(X) = c1(X) - s(X) * c0(X)                            |  |
|  |   Constant term = c1[0] (in coeff domain)                  |  |
|  +------------------------------------------------------------+  |
|                              |                                   |
|  +------------------------------------------------------------+  |
|  | Stage 2: Decompose c0 coefficients                         |  |
|  |   Grid: [N/256, B, 1]                                      |  |
|  |                                                            |  |
|  |   For coeff j = 0..N-1:                                    |  |
|  |     val = c0[j]                                            |  |
|  |     For level l = 0..L_ks-1:                               |  |
|  |       digits[j, l] = (val >> (l * base_log)) & mask        |  |
|  +------------------------------------------------------------+  |
|                              |                                   |
|  +------------------------------------------------------------+  |
|  | Stage 3: Accumulate key-switched LWE                       |  |
|  |   Grid: [n/256, B, 1]  Threads: [256, 1, 1]                |  |
|  |                                                            |  |
|  |   For output LWE coeff i = 0..n-1:                         |  |
|  |     sum = 0                                                |  |
|  |     For j = 0..N-1:                                        |  |
|  |       For l = 0..L_ks-1:                                   |  |
|  |         sum += digits[j,l] * ksk[j, l, i]                  |  |
|  |     lwe[b, i] = sum                                        |  |
|  |                                                            |  |
|  |   For body (i = n):                                        |  |
|  |     lwe[b, n] = c1[0] - sum_of_products                    |  |
|  +------------------------------------------------------------+  |
|                                                                  |
+------------------------------------------------------------------+
```

---

## 5. Parallelization Strategy

### 5.1 Thread Hierarchy

```
+------------------------------------------------------------------+
|                    PARALLELIZATION HIERARCHY                      |
+------------------------------------------------------------------+

Level 1: BATCH PARALLELISM (B)
    Each ciphertext in the batch is independent
    Typical B = 256-1024 for good GPU utilization

Level 2: COMPONENT PARALLELISM (2)
    RLWE has 2 polynomials (c0, c1)
    Both can be processed in parallel

Level 3: COEFFICIENT PARALLELISM (N)
    All N=1024 coefficients are independent for:
    - Pointwise operations
    - Decomposition
    - External product (in NTT domain)

Level 4: DECOMPOSITION PARALLELISM (L)
    L=4 decomposition levels can be parallelized
    But often serialized for register efficiency

TOTAL PARALLELISM: B * 2 * N = 256 * 2 * 1024 = 524,288 threads
(Well exceeds typical GPU thread count for full utilization)
```

### 5.2 Work Distribution

```cpp
// Kernel launch configurations for M1/M2/M3 GPUs
struct KernelConfig {
    // NTT kernel
    struct NTT {
        static constexpr int THREADS_X = 256;  // Per threadgroup
        static constexpr int THREADS_Y = 1;
        static constexpr int THREADS_Z = 1;
        
        // Grid size for B polynomials of size N
        static dim3 grid(int B, int N) {
            return {N / (2 * THREADS_X), B, 1};
        }
    };
    
    // External product kernel
    struct ExtProd {
        static constexpr int THREADS_X = 256;  // Coefficient dimension
        static constexpr int THREADS_Y = 1;    // Output component
        static constexpr int THREADS_Z = 1;    // Batch
        
        static dim3 grid(int B, int N) {
            return {(N + THREADS_X - 1) / THREADS_X, 2, B};
        }
    };
    
    // Blind rotation (single CMux step)
    struct CMux {
        static constexpr int THREADS_X = 256;
        static constexpr int THREADS_Y = 2;    // Both components
        static constexpr int THREADS_Z = 1;
        
        static dim3 grid(int B, int N) {
            return {(N + THREADS_X - 1) / THREADS_X, 1, B};
        }
    };
};
```

---

## 6. Memory Bandwidth Analysis

### 6.1 Theoretical Limits

```
Apple Silicon Memory Bandwidth:
  M1:     68 GB/s
  M1 Pro: 200 GB/s
  M1 Max: 400 GB/s
  M2:     100 GB/s
  M2 Pro: 200 GB/s
  M2 Max: 400 GB/s
  M3 Max: 400 GB/s
```

### 6.2 Per-Operation Memory Traffic

```
Operation           | Read (bytes)        | Write (bytes)       | Total
--------------------|---------------------|---------------------|--------
Init Accumulator    | N*8 (test poly)     | 2*N*8 (acc)         | 24 KB
Forward NTT         | 2*N*8 + N*8 (twid)  | 2*N*8               | 40 KB  
CMux (per step):    |                     |                     |
  - Rotate          | 2*N*8               | 2*N*8               | 32 KB
  - Decompose       | 2*N*8               | 2*L*N*8 (L=4)       | 80 KB
  - Ext Product     | 2*L*N*8 + 2*L*2*N*8 | 2*N*8               | 144 KB
  - SUBTOTAL/CMux   |                     |                     | 256 KB

Full Blind Rotation (n=512 CMux steps):
  - Per ciphertext: 256 KB * 512 = 128 MB
  - At 400 GB/s: 0.32 ms per ciphertext

Key Switching:
  - Decompose: N*8 + N*L_ks*8 = 40 KB (L_ks=4)
  - Accumulate: N*L_ks*(n+1)*8 + (n+1)*8 = 8.4 MB
  - Total: ~8.5 MB per ciphertext
  - At 400 GB/s: 0.02 ms per ciphertext
```

### 6.3 Optimization: Fused Kernel Memory Savings

```
UNFUSED (Current Implementation):
  - Multiple kernel launches
  - Full intermediate results written to global memory
  - Memory traffic: 256 KB per CMux step

FUSED (Proposed Implementation):
  - Single kernel for rotate + decompose + extprod
  - Digits kept in registers/threadgroup memory
  - Rotated values computed on-the-fly

FUSED Memory per CMux:
  - Read:  2*N*8 (acc) + 2*L*2*N*8 (bsk) = 144 KB
  - Write: 2*N*8 (acc) = 16 KB
  - Total: 160 KB (37% reduction)

Estimated speedup from fusion: 1.6x memory bandwidth reduction
```

---

## 7. Metal Shader Specifications

### 7.1 Modular Arithmetic Functions

```metal
// gpu_tfhe_kernels.metal

// Barrett reduction for modulus Q < 2^28
// Precompute: mu = floor(2^56 / Q)
inline uint64_t barrett_reduce(uint64_t x, uint64_t q, uint64_t mu) {
    uint64_t q_hat = (x * mu) >> 56;
    uint64_t r = x - q_hat * q;
    return r >= q ? r - q : r;
}

// Modular multiplication with Barrett reduction
inline uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t q, uint64_t mu) {
    // For a, b < Q < 2^28, product < 2^56
    uint64_t product = a * b;
    return barrett_reduce(product, q, mu);
}

// Montgomery multiplication (alternative for frequent multiplications)
inline uint64_t mont_mul(uint64_t a, uint64_t b, uint64_t q, uint64_t q_inv) {
    uint64_t lo = a * b;
    uint64_t hi = metal::mulhi(a, b);
    uint64_t m = (lo * q_inv) & 0xFFFFFFFF;
    uint64_t t = hi + metal::mulhi(m, q) + ((lo + m * q) < lo ? 1 : 0);
    return t >= q ? t - q : t;
}
```

### 7.2 Fused CMux Kernel

```metal
// Fused CMux kernel: rotate + decompose + external product
kernel void cmux_fused(
    device uint64_t* acc         [[buffer(0)]],  // [B, 2, N] in/out
    constant uint64_t* bsk       [[buffer(1)]],  // [2, L, 2, N] for current BK[i]
    constant int* rotations      [[buffer(2)]],  // [B] rotation amounts
    constant uint64_t& q         [[buffer(3)]],  // Modulus
    constant uint64_t& mu        [[buffer(4)]],  // Barrett constant
    constant uint& n_dim         [[buffer(5)]],  // N
    constant uint& l_dim         [[buffer(6)]],  // L
    constant uint& base_log      [[buffer(7)]],  // Decomposition base log
    
    uint3 gid [[thread_position_in_grid]],
    uint3 grid_size [[threads_per_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]]
) {
    uint batch_idx = gid.z;
    uint out_comp = gid.y;
    uint coeff_idx = gid.x;
    
    if (coeff_idx >= n_dim) return;
    
    int rotation = rotations[batch_idx];
    uint64_t mask = (1ULL << base_log) - 1;
    
    // Compute rotated coefficient indices
    int src_idx = (int)coeff_idx - rotation;
    bool negate = false;
    if (src_idx < 0) src_idx += 2 * n_dim;
    if (src_idx >= (int)n_dim) {
        src_idx -= n_dim;
        negate = true;
    }
    
    // Load accumulator values and compute diff = rotated - original
    uint64_t acc_orig[2];
    uint64_t acc_rot[2];
    uint64_t diff[2];
    
    for (int c = 0; c < 2; c++) {
        uint orig_idx = batch_idx * 2 * n_dim + c * n_dim + coeff_idx;
        uint rot_src = batch_idx * 2 * n_dim + c * n_dim + src_idx;
        
        acc_orig[c] = acc[orig_idx];
        acc_rot[c] = acc[rot_src];
        if (negate) acc_rot[c] = acc_rot[c] == 0 ? 0 : q - acc_rot[c];
        
        diff[c] = acc_rot[c] >= acc_orig[c] ? 
                  acc_rot[c] - acc_orig[c] : 
                  q - acc_orig[c] + acc_rot[c];
    }
    
    // Decompose diff into digits and accumulate external product
    uint64_t ext_prod = 0;
    
    for (int in_c = 0; in_c < 2; in_c++) {
        uint64_t val = diff[in_c];
        for (uint l = 0; l < l_dim; l++) {
            uint64_t digit = (val >> (l * base_log)) & mask;
            
            // Access BSK: bsk[in_c, l, out_comp, coeff]
            uint bsk_idx = in_c * l_dim * 2 * n_dim + 
                          l * 2 * n_dim + 
                          out_comp * n_dim + 
                          coeff_idx;
            uint64_t bsk_val = bsk[bsk_idx];
            
            ext_prod = mod_add(ext_prod, mod_mul(digit, bsk_val, q, mu), q);
        }
    }
    
    // Update accumulator: acc += ext_prod
    uint out_idx = batch_idx * 2 * n_dim + out_comp * n_dim + coeff_idx;
    acc[out_idx] = mod_add(acc_orig[out_comp], ext_prod, q);
}
```

---

## 8. Implementation Roadmap

### Phase 1: Core Kernels (Week 1-2)
1. Implement optimized NTT/INTT kernels with shared memory
2. Implement modular arithmetic using Barrett reduction
3. Add kernel unit tests

### Phase 2: Fused Operations (Week 3-4)
1. Implement fused CMux kernel
2. Implement key switching kernel
3. Benchmark individual kernels

### Phase 3: Full Pipeline (Week 5-6)
1. Integrate kernels into GPUTFHEEngine
2. Implement batch scheduling
3. End-to-end testing with circuit evaluation

### Phase 4: Optimization (Week 7-8)
1. Profile and identify bottlenecks
2. Tune threadgroup sizes and memory access patterns
3. Implement prefetching and double-buffering
4. Final benchmarks

---

## 9. Performance Targets

| Metric | Current (CPU Fallback) | Target (GPU Optimized) |
|--------|------------------------|------------------------|
| Single Bootstrap | ~500 ms | < 10 ms |
| Batch Bootstrap (256) | ~100 s | < 1 s |
| Throughput | ~2 ops/s | > 500 ops/s |
| Memory Efficiency | 100% (no reuse) | < 50% (fused) |

---

## 10. Appendix: MLX Integration Notes

### 10.1 MLX Array Requirements

```cpp
// MLX uses lazy evaluation - ensure explicit evaluation before Metal
mx::array result = kernel_output;
mx::eval(result);  // Force synchronous execution

// For custom Metal kernels, use mlx::core::metal::
// Note: MLX 0.x may require raw Metal integration
```

### 10.2 Custom Metal Kernel Registration

```cpp
// Register custom kernels with MLX metal backend
void registerTFHEKernels() {
    // Compile shader library
    auto library = mx::metal::compile_library(
        tfhe_kernels_source,  // Metal shader source string
        "tfhe_kernels"
    );
    
    // Register individual kernels
    mx::metal::register_kernel("ntt_forward", library);
    mx::metal::register_kernel("cmux_fused", library);
    mx::metal::register_kernel("key_switch", library);
}
```

---

## Document Version

- Version: 1.0
- Date: 2024
- Author: Architecture Team
- Status: Design Specification
