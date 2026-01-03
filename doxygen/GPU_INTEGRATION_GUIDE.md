# GPU Backend Integration Guide

**Moving GPU Acceleration from tfhe to fhe**

*Lux Industries Inc. | December 2024*

---

## Overview

This guide documents the migration of GPU-accelerated FHE operations from the high-level Go package (`~/work/lux/tfhe/gpu/`) to the low-level C++ library (`~/work/lux/fhe/`). This consolidation enables:

1. **Better performance**: Native C++ with CUDA/Metal instead of CGO overhead
2. **Unified build system**: Single CMake for all platforms
3. **Tighter integration**: Direct access to OpenFHE internals
4. **Multi-backend**: CUDA, Metal (MLX), and CPU fallback

---

## Current Architecture

### High-Level Go Package (`tfhe/gpu/`)

```
~/work/lux/tfhe/gpu/
├── blind_rotate.go       # GPU blind rotation
├── blind_rotate_stub.go  # CPU fallback
├── blind_rotate_test.go
├── bsk_cache.go          # Bootstrapping key cache
├── bsk_cache_stub.go
├── cmux.go               # Controlled MUX operations
├── cmux_stub.go
├── engine.go             # GPU engine abstraction
├── engine_stub.go
├── external_product.go   # External product computation
├── memory.go             # GPU memory management
├── memory_stub.go
├── mlx_ops.go            # Apple MLX operations
├── multigpu.go           # Multi-GPU distribution
├── multigpu_stub.go
├── ntt.go                # Number Theoretic Transform
├── ntt_test.go
├── scheduler.go          # Work scheduling
└── scheduler_stub.go
```

### Target Architecture (`fhe/src/gpu/`)

```
~/work/lux/fhe/src/gpu/
├── CMakeLists.txt        # GPU build config
├── cuda/
│   ├── blind_rotate.cu   # CUDA blind rotation
│   ├── cmux.cu           # CUDA CMux
│   ├── external_product.cu
│   ├── memory.cu         # CUDA memory management
│   ├── ntt.cu            # CUDA NTT kernels
│   └── scheduler.cu      # CUDA stream management
├── metal/
│   ├── blind_rotate.metal
│   ├── cmux.metal
│   ├── ntt.metal
│   └── ops.metal
├── common/
│   ├── engine.hpp        # Abstract engine interface
│   ├── memory.hpp        # Memory abstraction
│   └── scheduler.hpp     # Work scheduler
└── bindings/
    └── gpu_bindings.cpp  # C bindings for Go FFI
```

---

## Migration Plan

### Phase 1: Core NTT Kernels (Week 1-2)

The NTT (Number Theoretic Transform) is the performance bottleneck.

**CUDA Implementation**:

```cpp
// fhe/src/gpu/cuda/ntt.cu
#include <cuda_runtime.h>
#include "ntt.hpp"

namespace lux::fhe::gpu {

// Cooley-Tukey NTT on GPU
__global__ void ntt_forward_kernel(
    uint64_t* data,
    const uint64_t* twiddles,
    int n,
    int log_n,
    uint64_t modulus
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int stage = 0; stage < log_n; stage++) {
        int half_len = 1 << stage;
        int len = half_len << 1;
        int group = idx / half_len;
        int pos = idx % half_len;

        if (group < n / len) {
            int i = group * len + pos;
            int j = i + half_len;

            uint64_t w = twiddles[half_len + pos];
            uint64_t u = data[i];
            uint64_t v = mul_mod(data[j], w, modulus);

            data[i] = add_mod(u, v, modulus);
            data[j] = sub_mod(u, v, modulus);
        }

        __syncthreads();
    }
}

void ntt_forward(
    uint64_t* device_data,
    const NTTContext& ctx,
    cudaStream_t stream
) {
    int threads = min(ctx.n / 2, 1024);
    int blocks = (ctx.n / 2 + threads - 1) / threads;

    ntt_forward_kernel<<<blocks, threads, 0, stream>>>(
        device_data,
        ctx.device_twiddles,
        ctx.n,
        ctx.log_n,
        ctx.modulus
    );
}

} // namespace lux::fhe::gpu
```

**Metal Implementation**:

```metal
// fhe/src/gpu/metal/ntt.metal
#include <metal_stdlib>
using namespace metal;

kernel void ntt_forward(
    device uint64_t* data [[buffer(0)]],
    constant uint64_t* twiddles [[buffer(1)]],
    constant int& n [[buffer(2)]],
    constant int& log_n [[buffer(3)]],
    constant uint64_t& modulus [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    for (int stage = 0; stage < log_n; stage++) {
        int half_len = 1 << stage;
        int len = half_len << 1;
        int group = idx / half_len;
        int pos = idx % half_len;

        if (group < n / len) {
            int i = group * len + pos;
            int j = i + half_len;

            uint64_t w = twiddles[half_len + pos];
            uint64_t u = data[i];
            uint64_t v = mul_mod(data[j], w, modulus);

            data[i] = add_mod(u, v, modulus);
            data[j] = sub_mod(u, v, modulus);
        }

        threadgroup_barrier(mem_flags::mem_device);
    }
}
```

### Phase 2: Blind Rotation (Week 3-4)

Blind rotation is the core of bootstrapping.

```cpp
// fhe/src/gpu/cuda/blind_rotate.cu
namespace lux::fhe::gpu {

// Parallel CMux for blind rotation
__global__ void cmux_kernel(
    complex<double>* ct0_fft,
    complex<double>* ct1_fft,
    const complex<double>* gsw_fft,
    int n,
    int decomp_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // External product computation
    complex<double> sum0(0, 0);
    complex<double> sum1(0, 0);

    for (int i = 0; i < decomp_size; i++) {
        // ct0 contribution
        sum0 += ct0_fft[i * n + idx] * gsw_fft[i * 2 * n + idx];
        sum1 += ct0_fft[i * n + idx] * gsw_fft[(i * 2 + 1) * n + idx];

        // ct1 contribution
        sum0 += ct1_fft[i * n + idx] * gsw_fft[(decomp_size + i) * 2 * n + idx];
        sum1 += ct1_fft[i * n + idx] * gsw_fft[(decomp_size + i) * 2 + 1 * n + idx];
    }

    ct0_fft[idx] = sum0;
    ct1_fft[idx] = sum1;
}

void blind_rotate(
    Ciphertext& ct,
    const BootstrappingKey& bsk,
    const std::vector<int>& rotations,
    cudaStream_t stream
) {
    // Upload ciphertext to GPU
    auto ct_gpu = upload_ciphertext(ct, stream);

    // Apply each rotation
    for (size_t i = 0; i < rotations.size(); i++) {
        if (rotations[i] != 0) {
            // Polynomial rotation
            rotate_polynomial<<<blocks, threads, 0, stream>>>(
                ct_gpu, rotations[i]
            );

            // CMux with BSK[i]
            cmux_kernel<<<blocks, threads, 0, stream>>>(
                ct_gpu.c0_fft,
                ct_gpu.c1_fft,
                bsk.gsw_fft[i],
                ct.n,
                bsk.decomp_size
            );
        }
    }

    // Download result
    download_ciphertext(ct_gpu, ct, stream);
}

} // namespace lux::fhe::gpu
```

### Phase 3: Multi-GPU Support (Week 5-6)

```cpp
// fhe/src/gpu/common/multigpu.hpp
namespace lux::fhe::gpu {

class MultiGPUEngine {
private:
    std::vector<int> devices_;
    std::vector<cudaStream_t> streams_;
    size_t current_device_ = 0;

public:
    MultiGPUEngine() {
        int device_count;
        cudaGetDeviceCount(&device_count);

        for (int i = 0; i < device_count; i++) {
            devices_.push_back(i);
            cudaSetDevice(i);
            cudaStream_t stream;
            cudaStreamCreate(&stream);
            streams_.push_back(stream);
        }
    }

    // Distribute batch across GPUs
    std::vector<Ciphertext> batch_bootstrap(
        const std::vector<Ciphertext>& cts,
        const BootstrappingKey& bsk
    ) {
        int n = cts.size();
        int num_gpus = devices_.size();
        int per_gpu = (n + num_gpus - 1) / num_gpus;

        std::vector<std::future<std::vector<Ciphertext>>> futures;

        for (int g = 0; g < num_gpus; g++) {
            int start = g * per_gpu;
            int end = std::min(start + per_gpu, n);

            if (start < n) {
                futures.push_back(std::async([&, g, start, end]() {
                    cudaSetDevice(devices_[g]);
                    std::vector<Ciphertext> batch(
                        cts.begin() + start,
                        cts.begin() + end
                    );

                    for (auto& ct : batch) {
                        bootstrap_single(ct, bsk, streams_[g]);
                    }

                    cudaStreamSynchronize(streams_[g]);
                    return batch;
                }));
            }
        }

        // Collect results
        std::vector<Ciphertext> results;
        for (auto& f : futures) {
            auto batch = f.get();
            results.insert(results.end(), batch.begin(), batch.end());
        }

        return results;
    }
};

} // namespace lux::fhe::gpu
```

### Phase 4: Go Bindings (Week 7)

```cpp
// fhe/src/gpu/bindings/gpu_bindings.cpp
extern "C" {

#include "gpu_bindings.h"

// Engine management
void* lux_fhe_gpu_engine_create() {
    return new lux::fhe::gpu::Engine();
}

void lux_fhe_gpu_engine_destroy(void* engine) {
    delete static_cast<lux::fhe::gpu::Engine*>(engine);
}

// NTT operations
void lux_fhe_gpu_ntt_forward(
    void* engine,
    uint64_t* data,
    int n,
    uint64_t modulus
) {
    auto* eng = static_cast<lux::fhe::gpu::Engine*>(engine);
    eng->ntt_forward(data, n, modulus);
}

// Bootstrapping
void lux_fhe_gpu_bootstrap(
    void* engine,
    void* ciphertext,
    void* bsk
) {
    auto* eng = static_cast<lux::fhe::gpu::Engine*>(engine);
    auto* ct = static_cast<lux::fhe::Ciphertext*>(ciphertext);
    auto* key = static_cast<lux::fhe::BootstrappingKey*>(bsk);
    eng->bootstrap(*ct, *key);
}

// Batch operations
void lux_fhe_gpu_batch_bootstrap(
    void* engine,
    void** ciphertexts,
    int count,
    void* bsk
) {
    auto* eng = static_cast<lux::fhe::gpu::Engine*>(engine);
    auto* key = static_cast<lux::fhe::BootstrappingKey*>(bsk);

    std::vector<lux::fhe::Ciphertext*> cts(count);
    for (int i = 0; i < count; i++) {
        cts[i] = static_cast<lux::fhe::Ciphertext*>(ciphertexts[i]);
    }

    eng->batch_bootstrap(cts, *key);
}

} // extern "C"
```

**Go Wrapper**:

```go
// tfhe/gpu/engine.go (updated to use fhe bindings)
package gpu

/*
#cgo LDFLAGS: -L${SRCDIR}/../../fhe/build -llux_fhe_gpu
#cgo CFLAGS: -I${SRCDIR}/../../fhe/include

#include "gpu_bindings.h"
*/
import "C"
import "unsafe"

type Engine struct {
    ptr unsafe.Pointer
}

func NewEngine() *Engine {
    return &Engine{
        ptr: C.lux_fhe_gpu_engine_create(),
    }
}

func (e *Engine) Close() {
    C.lux_fhe_gpu_engine_destroy(e.ptr)
}

func (e *Engine) Bootstrap(ct *Ciphertext, bsk *BootstrappingKey) {
    C.lux_fhe_gpu_bootstrap(e.ptr, ct.ptr, bsk.ptr)
}

func (e *Engine) BatchBootstrap(cts []*Ciphertext, bsk *BootstrappingKey) {
    ptrs := make([]unsafe.Pointer, len(cts))
    for i, ct := range cts {
        ptrs[i] = ct.ptr
    }

    C.lux_fhe_gpu_batch_bootstrap(
        e.ptr,
        (*unsafe.Pointer)(&ptrs[0]),
        C.int(len(cts)),
        bsk.ptr,
    )
}
```

---

## CMake Configuration

```cmake
# fhe/src/gpu/CMakeLists.txt

# Find CUDA
find_package(CUDA 11.0)

# Find Metal (macOS)
if(APPLE)
    find_library(METAL_FRAMEWORK Metal)
    find_library(METALKIT_FRAMEWORK MetalKit)
    find_library(FOUNDATION_FRAMEWORK Foundation)
endif()

# GPU sources
set(GPU_COMMON_SOURCES
    common/engine.cpp
    common/memory.cpp
    common/scheduler.cpp
)

# CUDA backend
if(CUDA_FOUND)
    enable_language(CUDA)

    set(CUDA_SOURCES
        cuda/ntt.cu
        cuda/blind_rotate.cu
        cuda/cmux.cu
        cuda/external_product.cu
        cuda/memory.cu
    )

    cuda_add_library(lux_fhe_gpu_cuda STATIC ${CUDA_SOURCES})
    target_compile_options(lux_fhe_gpu_cuda PRIVATE
        -arch=sm_70  # Volta and newer
        --use_fast_math
        -O3
    )
endif()

# Metal backend (macOS)
if(APPLE AND METAL_FRAMEWORK)
    set(METAL_SOURCES
        metal/ntt.metal
        metal/blind_rotate.metal
        metal/cmux.metal
    )

    # Compile Metal shaders
    foreach(METAL_FILE ${METAL_SOURCES})
        get_filename_component(METAL_NAME ${METAL_FILE} NAME_WE)
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${METAL_NAME}.air
            COMMAND xcrun -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/${METAL_FILE}
                    -o ${CMAKE_CURRENT_BINARY_DIR}/${METAL_NAME}.air
            DEPENDS ${METAL_FILE}
        )
        list(APPEND METAL_AIR_FILES ${CMAKE_CURRENT_BINARY_DIR}/${METAL_NAME}.air)
    endforeach()

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/lux_fhe.metallib
        COMMAND xcrun -sdk macosx metallib ${METAL_AIR_FILES}
                -o ${CMAKE_CURRENT_BINARY_DIR}/lux_fhe.metallib
        DEPENDS ${METAL_AIR_FILES}
    )
endif()

# Combined library
add_library(lux_fhe_gpu SHARED
    ${GPU_COMMON_SOURCES}
    bindings/gpu_bindings.cpp
)

if(CUDA_FOUND)
    target_link_libraries(lux_fhe_gpu lux_fhe_gpu_cuda)
    target_compile_definitions(lux_fhe_gpu PRIVATE LUX_FHE_CUDA)
endif()

if(APPLE AND METAL_FRAMEWORK)
    target_link_libraries(lux_fhe_gpu
        ${METAL_FRAMEWORK}
        ${METALKIT_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
    )
    target_compile_definitions(lux_fhe_gpu PRIVATE LUX_FHE_METAL)
endif()

target_include_directories(lux_fhe_gpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
```

---

## Build Instructions

### Linux (CUDA)

```bash
cd ~/work/lux/fhe
mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_GPU=ON \
    -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;90"

make -j$(nproc) lux_fhe_gpu
```

### macOS (Metal)

```bash
cd ~/work/lux/fhe
mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_GPU=ON \
    -DWITH_METAL=ON

make -j$(sysctl -n hw.ncpu) lux_fhe_gpu
```

### Testing

```bash
# Run GPU tests
ctest -R gpu

# Benchmark
./build/bin/fhe_gpu_benchmark --iterations 1000
```

---

## Performance Comparison

### Expected Results

| Operation | Go (CGO) | C++ CUDA | Improvement |
|-----------|----------|----------|-------------|
| NTT (N=16384) | 2ms | 50μs | **40x** |
| Bootstrap | 10ms | 2ms | **5x** |
| CMux | 500μs | 50μs | **10x** |
| Batch (100 CTs) | 800ms | 40ms | **20x** |

### Overhead Breakdown

| Layer | Latency |
|-------|---------|
| Go → CGO | 10μs |
| CGO → C++ | 1μs |
| C++ → CUDA kernel | 5μs |
| CUDA execution | Variable |
| Total overhead | ~16μs |

---

## Migration Checklist

- [ ] **Phase 1**: NTT kernels (CUDA + Metal)
- [ ] **Phase 2**: Blind rotation kernels
- [ ] **Phase 3**: Multi-GPU engine
- [ ] **Phase 4**: C bindings
- [ ] **Phase 5**: Update Go package to use new bindings
- [ ] **Phase 6**: CI/CD for GPU builds
- [ ] **Phase 7**: Benchmarks and validation
- [ ] **Phase 8**: Documentation and examples

---

## Files to Migrate

| Current Location | New Location | Status |
|-----------------|--------------|--------|
| `tfhe/gpu/ntt.go` | `fhe/src/gpu/cuda/ntt.cu` | Pending |
| `tfhe/gpu/blind_rotate.go` | `fhe/src/gpu/cuda/blind_rotate.cu` | Pending |
| `tfhe/gpu/cmux.go` | `fhe/src/gpu/cuda/cmux.cu` | Pending |
| `tfhe/gpu/engine.go` | `fhe/src/gpu/common/engine.cpp` | Pending |
| `tfhe/gpu/memory.go` | `fhe/src/gpu/cuda/memory.cu` | Pending |
| `tfhe/gpu/scheduler.go` | `fhe/src/gpu/common/scheduler.cpp` | Pending |
| `tfhe/gpu/multigpu.go` | `fhe/src/gpu/common/multigpu.cpp` | Pending |
| `tfhe/gpu/mlx_ops.go` | `fhe/src/gpu/metal/ops.metal` | Pending |

---

## Contact

- **Technical Lead**: oss@lux.network
- **GPU Team**: gpu@lux.network

---

*Confidential - For Internal Use Only*

© 2024-2025 Lux Industries Inc. All rights reserved.
