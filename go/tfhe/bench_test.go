// OpenFHE CGO Benchmark Tests
// Run with: CGO_CXXFLAGS='-I.../openfhe/include -I.../openfhe/core' CGO_LDFLAGS='-L.../lib -lOPENFHEbinfhe -lOPENFHEcore' go test -bench=. -benchmem

package tfhe

import (
	"testing"
)

var (
	benchCtx *Context
	benchSK  *SecretKey
	benchCT1 *Ciphertext
	benchCT2 *Ciphertext
)

func setupOpenFHEBench(b *testing.B) {
	if benchCtx != nil {
		return
	}
	benchCtx = NewContext(STD128)
	benchSK = benchCtx.KeyGen()
	benchCtx.BootstrapKeyGen(benchSK) // Correct method name
	benchCT1 = benchCtx.Encrypt(benchSK, true)
	benchCT2 = benchCtx.Encrypt(benchSK, false)
}

// Key generation benchmarks
func BenchmarkOpenFHEKeyGen(b *testing.B) {
	ctx := NewContext(STD128)
	defer ctx.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ctx.KeyGen()
		// Finalizer will clean up
	}
}

func BenchmarkOpenFHEBootstrapKeyGen(b *testing.B) {
	ctx := NewContext(STD128)
	defer ctx.Close()
	sk := ctx.KeyGen()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctx.BootstrapKeyGen(sk)
	}
}

// Encryption benchmarks
func BenchmarkOpenFHEEncrypt(b *testing.B) {
	setupOpenFHEBench(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchCtx.Encrypt(benchSK, true)
		// Finalizer will clean up
	}
}

func BenchmarkOpenFHEDecrypt(b *testing.B) {
	setupOpenFHEBench(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchCtx.Decrypt(benchSK, benchCT1)
	}
}

// Gate benchmarks
func BenchmarkOpenFHEAND(b *testing.B) {
	setupOpenFHEBench(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchCtx.AND(benchCT1, benchCT2)
	}
}

func BenchmarkOpenFHEOR(b *testing.B) {
	setupOpenFHEBench(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchCtx.OR(benchCT1, benchCT2)
	}
}

func BenchmarkOpenFHEXOR(b *testing.B) {
	setupOpenFHEBench(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchCtx.XOR(benchCT1, benchCT2)
	}
}

func BenchmarkOpenFHENOT(b *testing.B) {
	setupOpenFHEBench(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchCtx.NOT(benchCT1)
	}
}

func BenchmarkOpenFHENAND(b *testing.B) {
	setupOpenFHEBench(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchCtx.NAND(benchCT1, benchCT2)
	}
}

func BenchmarkOpenFHENOR(b *testing.B) {
	setupOpenFHEBench(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchCtx.NOR(benchCT1, benchCT2)
	}
}

func BenchmarkOpenFHEXNOR(b *testing.B) {
	setupOpenFHEBench(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchCtx.XNOR(benchCT1, benchCT2)
	}
}
