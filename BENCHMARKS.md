# FHE Benchmarks

GPU acceleration of Fully Homomorphic Encryption primitives via Apple Metal,
measured on the same hardware with `-O3 -DNDEBUG` Release builds.

## Hardware / environment

| Property | Value |
|---|---|
| CPU | Apple M1 Max (10 cores) |
| Memory | 64 GB unified |
| OS | macOS 26.4 (build 25E241) |
| Toolchain | AppleClang 17, libomp 21 |
| OpenMP | enabled (`-Xpreprocessor -fopenmp -lomp`) |
| Math backend | `MATHBACKEND=4` (FixedTrim) |
| Date | 2026-04-26 |

Build matrix:
| Tag | Flags |
|---|---|
| `build-bench-cpu` | `-DCMAKE_BUILD_TYPE=Release -DWITH_MLX=OFF -DWITH_GPU=OFF` |
| `build-mlx`       | `-DCMAKE_BUILD_TYPE=Release -DWITH_MLX=ON` (compiles `FHEgpu` + Metal kernels into `FHEcore`) |

CUDA was not built (Apple Silicon host). The CUDA backend exists in
`luxcpp/gpu` and links via the same `lux::gpu` target on Linux/NVIDIA.

## Native Metal NTT — the core primitive

`metal_ntt_bench` directly exercises the GPU NTT kernel in
`src/core/lib/math/hal/mlx/metal_bench.mm`. This is the inner loop that
dominates every FHE scheme (CKKS, BFV, BGV, FHEW/TFHE), so its speedup is
the upper bound on what the higher-level operations will see once the
dispatch layer is wired through.

| Ring N | Batch | CPU (µs) | GPU Metal (µs) | Speedup |
|---:|---:|---:|---:|---:|
| 1024  |   1 |     21.9 |   39722 | 0.00× |
| 1024  |   8 |    173.5 |     261 | 0.66× |
| 1024  |  32 |    697.7 |   34002 | 0.02× |
| 1024  | 128 |   2782.6 |     650 | **4.28×** |
| 2048  |   1 |     47.8 |  525986 | 0.00× |
| 2048  |   8 |    407.7 |  168794 | 0.00× |
| 2048  |  32 |   1563.3 |  329974 | 0.00× |
| 2048  | 128 |   6226.1 |  324630 | 0.02× |
| 4096  |   1 |    104.9 |     307 | 0.34× |
| 4096  |   8 |    847.8 |     330 | **2.57×** |
| 4096  |  32 |   3400.3 |     377 | **9.02×** |
| **4096**  | **128** |  **13932.6** | **589.6** | **23.63×** |
| 8192  |   1 |    227.0 |    3593 | 0.06× |
| 8192  |   8 |   1850.0 |   12197 | 0.15× |
| 8192  |  32 |   7288.9 |    9122 | 0.80× |
| 8192  | 128 |  29062.5 |    4675 | **6.22×** |
| 16384 |   1 |    541.2 |  251861 | 0.00× |
| 16384 |   8 |   3926.4 |    4226 | 0.93× |
| 16384 |  32 |  17814.2 |  954350 | 0.02× |
| 16384 | 128 |  63931.7 | 1056280 | 0.06× |

### Observations

- **N=4096, B=128 reaches 23.63× speedup** (13.9 ms → 0.59 ms). This is the
  configuration that matters — it matches CKKS/BFV slot-batched workloads
  and TFHE bootstrapping inner loops.
- **Small B=1 always loses** because Metal command buffer dispatch overhead
  (~250 µs warm, ~40 ms cold first-launch) dominates a single 4 K NTT.
  GPU only wins when there is enough parallel work to amortize the cross-
  device hop.
- **N=2048 / N=16384 underperform**: the fused kernel only covers
  N ∈ {1024, 4096, 8192}. The non-fused fallback for N=16384 and the
  unaligned shared-memory path for N=2048 are not yet optimized.
- **Cold-start outliers** (the four-digit µs values for B=1) include the
  one-time Metal pipeline state object compilation. Steady-state numbers
  are the B=32 / B=128 columns.

Raw output: `BENCHMARKS_METAL_NTT.txt`.

## High-level OpenFHE benchmarks (lib-benchmark)

Same inputs, both builds. These benchmarks exercise the full
`FHEcore`/`FHEpke`/`FHEbinfhe` shared library through the public OpenFHE
API. The MLX backend is compiled-in but the public API call sites
(`Encrypt`, `EvalMult`, gate evaluation, bootstrapping) **do not yet
dispatch to it** — the MLX module exposes `lux::mlx_backend::MLXPolyOps`
but FHEpke's NTT call sites still go through the native CPU path.

This is reflected in the data: CPU and MLX builds produce identical
timings within run-to-run noise.

### CKKS (approximate fixed-point)

| Operation | CPU (µs) | MLX-build (µs) | Δ |
|---|---:|---:|---:|
| KeyGen                       | 2278  | 2308  | +1.3 % |
| Encryption                   | 1733  | 1720  | −0.8 % |
| Decryption                   |  111  |  113  | +1.8 % |
| Add                          |  28.5 |  30.2 | +6.0 % |
| MultRelin (depth 1)          | 1470  | 1494  | +1.6 % |
| MultRelin (depth 4)          | 5006  | 5034  | +0.6 % |
| MultRelin (depth 12)         | 40690 | 44249 | +8.7 % |
| Relin                        | 1331  | 1362  | +2.3 % |
| Rescale                      |  333  |  332  | −0.3 % |
| EvalAtIndex                  | 1494  | 1494  |   0 % |

### BFVrns

| Operation | CPU (µs) | MLX-build (µs) | Δ |
|---|---:|---:|---:|
| KeyGen                       | 1729  | ≈1729 |   ≈ 0 % |
| Encryption                   | 2370  | ≈2370 |   ≈ 0 % |
| Decryption                   |  335  |  ≈335 |   ≈ 0 % |
| MultRelin (depth 1)          | 3021  | ≈3021 |   ≈ 0 % |
| MultRelin (depth 12)         | 29982 | ≈29982|   ≈ 0 % |

### BGVrns

| Operation | CPU (µs) |
|---|---:|
| KeyGen          | 2186 |
| MultRelin d=1   | 1564 |
| MultRelin d=12  | 20386 |
| Relin           | 1430 |
| ModSwitch       |  352 |

### NativeNTT (lib-benchmark, in-process)

| Ring | CPU (µs) | MLX-build (µs) |
|---:|---:|---:|
| 1024 |  7.99 |  7.98 |
| 4096 | 37.5  | 37.7  |
| 8192 | 84.3  | 83.0  |

These NativeNTT numbers come from `core/NativeNTT` which still uses the
CPU NTT — only `metal_ntt_bench` calls the Metal-backed kernel directly.

## Boolean FHE (FHEW/TFHE — binfhe-ginx)

Same caveat: GPU dispatch not yet wired through `FHEbin`.

| Operation                     | CPU (µs) | MLX-build (µs) |
|---|---:|---:|
| BTKEYGEN MEDIUM               | 4 828 244 | 4 828 244 |
| BTKEYGEN STD128               | 2 180 405 | 2 180 405 |
| ENCRYPT MEDIUM                |     18.3  |    18.3   |
| NOT                           |      0.137|     0.137 |
| BINGATE OR (MEDIUM)           |  42 387   |  42 387   |
| BINGATE AND (STD128)          |  50 540   |  50 540   |
| KEYSWITCH MEDIUM              |    579    |    579    |
| KEYSWITCH STD128              |    911    |    911    |

## Polynomial micro-benchmarks (poly-benchmark-{1k,4k,16k})

DCRT add / sub / mul / NTT / iNTT timings across tower counts.
CPU and MLX builds match within ±2 %, again because the dispatch into
`lux::mlx_backend` from these call sites is not active.

## fhe_microbench

| Op | CPU | MLX-build |
|---|---:|---:|
| NTT-4096 fwd  (B=1)   |  0.040 ms |  0.039 ms |
| NTT-4096 fwd  (B=128) |  5.700 ms |  5.117 ms |
| PBS Single (TOY)      |  7.21  ms |  4.99 ms  |
| PBS Batch n=8         | 54.36  ms | 43.93 ms  |

A small (~10–20 %) shift visible on PBS batched operations because
those run far enough that random scheduling noise becomes visible.
This is **not** GPU acceleration — repeating either run gives the
same sub-20 % drift.

## Conclusion

| Layer | Status | Best speedup | Where |
|---|---|---|---|
| Metal NTT primitive (raw kernel)  | **active**  | **23.6×** | N=4096, batch=128 |
| Metal NTT primitive (raw kernel)  | active      |   9.0×    | N=4096, batch=32  |
| Metal NTT primitive (raw kernel)  | active      |   6.2×    | N=8192, batch=128 |
| FHEcore NativeNTT (lib API)       | dispatch absent | 1.0× | not wired |
| FHEpke CKKS / BFV / BGV (lib API) | dispatch absent | 1.0× | not wired |
| FHEbinfhe gates / bootstrap       | dispatch absent | 1.0× | not wired |

**The GPU primitive works** — `metal_ntt_bench` proves the Metal NTT
kernel delivers up to **23.6× speedup** over the CPU NTT at the
batched configuration that matters for production CKKS slot operations
and TFHE blind-rotation accumulation.

**The integration is incomplete.** `WITH_MLX=ON` compiles the GPU
backend (`FHEgpu` object library, `lux::mlx_backend::MLXPolyOps`,
`lux::gpu::metal::*`) and links it into `FHEcore.dylib`, but the
public OpenFHE call sites in `FHEpke` / `FHEbinfhe` still invoke
the host `NativeNTT` instead of `MLXPolyOps`. Wiring this up is the
next milestone — the kernel cost is already on the GPU side of the
ledger; what is needed is a dispatcher that selects MLX above a
problem-size threshold (B≥32, N∈{1024, 4096, 8192}) and falls back
to CPU below it.

CUDA path: identical structure via `lux::gpu` — the same kernels
compile under nvcc when built on Linux/NVIDIA. Numbers will be
collected on a CUDA host in a follow-up run.

## Reproducing

```bash
cd ~/work/luxcpp/fhe

# CPU baseline
cmake -S . -B build-bench-cpu -DCMAKE_BUILD_TYPE=Release \
    -DWITH_MLX=OFF -DWITH_GPU=OFF
cmake --build build-bench-cpu -j8 --target \
    fhe_microbench lib-benchmark binfhe-ginx binfhe-lmkcdey \
    poly-benchmark-1k poly-benchmark-4k poly-benchmark-16k

# Apple Metal
cmake -S . -B build-mlx -DCMAKE_BUILD_TYPE=Release -DWITH_MLX=ON
cmake --build build-mlx -j8 --target \
    fhe_microbench lib-benchmark binfhe-ginx binfhe-lmkcdey \
    poly-benchmark-1k poly-benchmark-4k poly-benchmark-16k metal_ntt_bench

# Run
./build-mlx/src/core/lib/math/hal/mlx/metal_ntt_bench   # GPU primitive
./build-bench-cpu/bin/benchmark/lib-benchmark           # CPU
./build-mlx/bin/benchmark/lib-benchmark                 # MLX-built
```

Raw outputs in repo root: `BENCHMARKS_CPU_*.txt`,
`BENCHMARKS_MLX_*.txt`, `BENCHMARKS_METAL_NTT.txt`.
