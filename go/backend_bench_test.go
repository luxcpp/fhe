// FHE Backend Comparison Benchmarks
// Compares CPU, MLX (Metal GPU), and WebGPU backends
//
// Run with specific backend:
//   CGO_LDFLAGS="-L../build-local/lib" go test -bench=. -tags=cpu
//   CGO_LDFLAGS="-L../build_mlx/lib" go test -bench=. -tags=mlx
//   CGO_LDFLAGS="-L../build-webgpu/lib" go test -bench=. -tags=webgpu
//
// Or run all available:
//   go test -bench=. -benchmem -benchtime=3s ./...

package main

import (
	"crypto/rand"
	"fmt"
	"os"
	"runtime"
	"testing"
	"time"
)

// =============================================================================
// Backend Detection
// =============================================================================

type BackendInfo struct {
	Name      string
	Available bool
	Device    string
	Memory    uint64
}

func detectBackend() BackendInfo {
	// Check environment for backend hint
	backend := os.Getenv("FHE_BACKEND")

	info := BackendInfo{
		Name:      "CPU",
		Available: true,
		Device:    runtime.GOARCH,
	}

	switch backend {
	case "mlx", "metal":
		info.Name = "MLX (Metal)"
		info.Device = "Apple GPU"
	case "webgpu", "dawn":
		info.Name = "WebGPU"
		info.Device = "GPU (WebGPU)"
	case "cuda":
		info.Name = "CUDA"
		info.Device = "NVIDIA GPU"
	default:
		// Check if MLX libraries are available
		if _, err := os.Stat("../build_mlx/lib/libOPENFHEcore.dylib"); err == nil {
			info.Name = "MLX (Metal)"
			info.Device = "Apple GPU"
		}
	}

	return info
}

// =============================================================================
// NTT Benchmarks (Core operation for lattice crypto)
// =============================================================================

func BenchmarkNTT(b *testing.B) {
	backend := detectBackend()
	b.Logf("Backend: %s (%s)", backend.Name, backend.Device)

	sizes := []int{1024, 4096, 8192, 16384, 32768}

	for _, n := range sizes {
		data := make([]uint64, n)
		for i := range data {
			data[i] = uint64(i)
		}

		b.Run(fmt.Sprintf("%s/n=%d", backend.Name, n), func(b *testing.B) {
			b.SetBytes(int64(n * 8))
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				// Simulated NTT - in real implementation this calls the backend
				nttForward(data)
			}
		})
	}
}

func BenchmarkINTT(b *testing.B) {
	backend := detectBackend()

	sizes := []int{1024, 4096, 8192, 16384}

	for _, n := range sizes {
		data := make([]uint64, n)
		for i := range data {
			data[i] = uint64(i)
		}

		b.Run(fmt.Sprintf("%s/n=%d", backend.Name, n), func(b *testing.B) {
			b.SetBytes(int64(n * 8))
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				nttInverse(data)
			}
		})
	}
}

// =============================================================================
// TFHE Gate Benchmarks
// =============================================================================

func BenchmarkTFHE_AND(b *testing.B) {
	backend := detectBackend()
	b.Run(backend.Name, func(b *testing.B) {
		// Setup: create encrypted bits
		ct1 := encryptBit(true)
		ct2 := encryptBit(false)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = evalAND(ct1, ct2)
		}
	})
}

func BenchmarkTFHE_NAND(b *testing.B) {
	backend := detectBackend()
	b.Run(backend.Name, func(b *testing.B) {
		ct1 := encryptBit(true)
		ct2 := encryptBit(true)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = evalNAND(ct1, ct2)
		}
	})
}

func BenchmarkTFHE_Bootstrap(b *testing.B) {
	backend := detectBackend()
	b.Run(backend.Name, func(b *testing.B) {
		ct := encryptBit(true)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = bootstrap(ct)
		}
	})
}

// =============================================================================
// CKKS Benchmarks (Approximate arithmetic)
// =============================================================================

func BenchmarkCKKS_Encrypt(b *testing.B) {
	backend := detectBackend()

	sizes := []int{1024, 4096, 8192}

	for _, n := range sizes {
		values := make([]float64, n)
		for i := range values {
			values[i] = float64(i) * 0.001
		}

		b.Run(fmt.Sprintf("%s/slots=%d", backend.Name, n), func(b *testing.B) {
			b.SetBytes(int64(n * 8))
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				_ = ckksEncrypt(values)
			}
		})
	}
}

func BenchmarkCKKS_Add(b *testing.B) {
	backend := detectBackend()
	b.Run(backend.Name, func(b *testing.B) {
		values := make([]float64, 4096)
		ct1 := ckksEncrypt(values)
		ct2 := ckksEncrypt(values)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = ckksAdd(ct1, ct2)
		}
	})
}

func BenchmarkCKKS_Mult(b *testing.B) {
	backend := detectBackend()
	b.Run(backend.Name, func(b *testing.B) {
		values := make([]float64, 4096)
		ct1 := ckksEncrypt(values)
		ct2 := ckksEncrypt(values)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = ckksMult(ct1, ct2)
		}
	})
}

func BenchmarkCKKS_Rotate(b *testing.B) {
	backend := detectBackend()

	rotations := []int{1, 4, 16, 64}

	for _, rot := range rotations {
		b.Run(fmt.Sprintf("%s/rot=%d", backend.Name, rot), func(b *testing.B) {
			values := make([]float64, 4096)
			ct := ckksEncrypt(values)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = ckksRotate(ct, rot)
			}
		})
	}
}

func BenchmarkCKKS_Bootstrap(b *testing.B) {
	backend := detectBackend()
	b.Run(backend.Name, func(b *testing.B) {
		values := make([]float64, 4096)
		ct := ckksEncrypt(values)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = ckksBootstrap(ct)
		}
	})
}

// =============================================================================
// Threshold FHE Benchmarks (Multi-party)
// =============================================================================

func BenchmarkThreshold_PartialDecrypt(b *testing.B) {
	backend := detectBackend()

	batchSizes := []int{10, 100, 1000}

	for _, batch := range batchSizes {
		b.Run(fmt.Sprintf("%s/batch=%d", backend.Name, batch), func(b *testing.B) {
			// Create batch of ciphertexts
			cts := make([][]byte, batch)
			for i := range cts {
				cts[i] = make([]byte, 1024)
				rand.Read(cts[i])
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = batchPartialDecrypt(cts)
			}
		})
	}
}

func BenchmarkThreshold_Combine(b *testing.B) {
	backend := detectBackend()

	parties := []int{3, 5, 7, 10}

	for _, n := range parties {
		b.Run(fmt.Sprintf("%s/parties=%d", backend.Name, n), func(b *testing.B) {
			partials := make([][]byte, n)
			for i := range partials {
				partials[i] = make([]byte, 256)
				rand.Read(partials[i])
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = combinePartials(partials)
			}
		})
	}
}

// =============================================================================
// Memory Bandwidth Benchmarks
// =============================================================================

func BenchmarkMemoryBandwidth(b *testing.B) {
	backend := detectBackend()

	sizes := []int{1 << 20, 4 << 20, 16 << 20, 64 << 20} // 1MB to 64MB

	for _, size := range sizes {
		src := make([]byte, size)
		dst := make([]byte, size)
		rand.Read(src)

		sizeMB := size / (1 << 20)
		b.Run(fmt.Sprintf("%s/%dMB", backend.Name, sizeMB), func(b *testing.B) {
			b.SetBytes(int64(size))

			for i := 0; i < b.N; i++ {
				copy(dst, src)
			}
		})
	}
}

// =============================================================================
// Comparison Report
// =============================================================================

func TestPrintBackendComparison(t *testing.T) {
	backend := detectBackend()

	fmt.Println()
	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║              FHE Backend Benchmark Comparison                  ║")
	fmt.Println("╠════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║ Backend:    %-50s ║\n", backend.Name)
	fmt.Printf("║ Device:     %-50s ║\n", backend.Device)
	fmt.Printf("║ Platform:   %-50s ║\n", runtime.GOOS+"/"+runtime.GOARCH)
	fmt.Printf("║ CPUs:       %-50d ║\n", runtime.NumCPU())
	fmt.Printf("║ Time:       %-50s ║\n", time.Now().Format(time.RFC3339))
	fmt.Println("╠════════════════════════════════════════════════════════════════╣")
	fmt.Println("║ Run benchmarks with:                                           ║")
	fmt.Println("║   go test -bench=. -benchmem -benchtime=3s                     ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Print expected performance characteristics
	fmt.Println("Expected Performance Characteristics:")
	fmt.Println("┌────────────────┬─────────┬─────────┬─────────┐")
	fmt.Println("│ Operation      │   CPU   │   MLX   │ WebGPU  │")
	fmt.Println("├────────────────┼─────────┼─────────┼─────────┤")
	fmt.Println("│ NTT (n=8192)   │  ~5ms   │  ~1ms   │  ~2ms   │")
	fmt.Println("│ TFHE AND gate  │ ~10ms   │  ~3ms   │  ~5ms   │")
	fmt.Println("│ CKKS Mult      │  ~8ms   │  ~2ms   │  ~4ms   │")
	fmt.Println("│ Bootstrap      │~300ms   │ ~80ms   │~120ms   │")
	fmt.Println("│ Batch (1000)   │~500ms   │~100ms   │~150ms   │")
	fmt.Println("└────────────────┴─────────┴─────────┴─────────┘")
	fmt.Println()
	fmt.Println("Note: Actual performance depends on hardware and workload.")
}

// =============================================================================
// Stub implementations (replace with actual CGO calls)
// =============================================================================

// NTT stubs
func nttForward(data []uint64) {
	// Placeholder - actual implementation calls C library
	_ = data
}

func nttInverse(data []uint64) {
	_ = data
}

// TFHE stubs
type tfheCiphertext struct{ data []byte }

func encryptBit(b bool) *tfheCiphertext {
	ct := &tfheCiphertext{data: make([]byte, 512)}
	rand.Read(ct.data)
	return ct
}

func evalAND(ct1, ct2 *tfheCiphertext) *tfheCiphertext {
	return &tfheCiphertext{data: make([]byte, 512)}
}

func evalNAND(ct1, ct2 *tfheCiphertext) *tfheCiphertext {
	return &tfheCiphertext{data: make([]byte, 512)}
}

func bootstrap(ct *tfheCiphertext) *tfheCiphertext {
	// Simulate bootstrap latency
	time.Sleep(time.Microsecond * 100)
	return &tfheCiphertext{data: make([]byte, 512)}
}

// CKKS stubs
type ckksCiphertext struct{ data []byte }

func ckksEncrypt(values []float64) *ckksCiphertext {
	ct := &ckksCiphertext{data: make([]byte, len(values)*16)}
	rand.Read(ct.data)
	return ct
}

func ckksAdd(ct1, ct2 *ckksCiphertext) *ckksCiphertext {
	return &ckksCiphertext{data: make([]byte, len(ct1.data))}
}

func ckksMult(ct1, ct2 *ckksCiphertext) *ckksCiphertext {
	return &ckksCiphertext{data: make([]byte, len(ct1.data))}
}

func ckksRotate(ct *ckksCiphertext, steps int) *ckksCiphertext {
	return &ckksCiphertext{data: make([]byte, len(ct.data))}
}

func ckksBootstrap(ct *ckksCiphertext) *ckksCiphertext {
	time.Sleep(time.Microsecond * 500)
	return &ckksCiphertext{data: make([]byte, len(ct.data))}
}

// Threshold stubs
func batchPartialDecrypt(cts [][]byte) [][]byte {
	results := make([][]byte, len(cts))
	for i := range results {
		results[i] = make([]byte, 32)
	}
	return results
}

func combinePartials(partials [][]byte) []byte {
	return make([]byte, 32)
}
