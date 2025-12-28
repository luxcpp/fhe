// Package tfhe provides Go bindings for TFHE (Fast Fully Homomorphic Encryption).
//
// TFHE enables boolean circuit evaluation on encrypted data with fast bootstrapping.
// Each gate operation takes ~10ms and refreshes the ciphertext noise.
package tfhe

/*
#cgo CXXFLAGS: -std=c++17
#cgo LDFLAGS: -lOpenFHEbinfhe -lOpenFHEcore -lOpenFHEpke

#include <stdlib.h>

// Forward declarations for OpenFHE types
typedef void* BinFHEContext;
typedef void* LWESecretKey;
typedef void* LWECiphertext;

// Context functions
BinFHEContext NewBinFHEContext();
void FreeBinFHEContext(BinFHEContext ctx);
void GenerateBinFHEContext(BinFHEContext ctx, int securityLevel, int method);

// Key functions
LWESecretKey KeyGen(BinFHEContext ctx);
void FreeLWESecretKey(LWESecretKey sk);
void BTKeyGen(BinFHEContext ctx, LWESecretKey sk);

// Encryption/Decryption
LWECiphertext Encrypt(BinFHEContext ctx, LWESecretKey sk, int plaintext);
int Decrypt(BinFHEContext ctx, LWESecretKey sk, LWECiphertext ct);
void FreeLWECiphertext(LWECiphertext ct);

// Gates
LWECiphertext EvalAND(BinFHEContext ctx, LWECiphertext ct1, LWECiphertext ct2);
LWECiphertext EvalOR(BinFHEContext ctx, LWECiphertext ct1, LWECiphertext ct2);
LWECiphertext EvalXOR(BinFHEContext ctx, LWECiphertext ct1, LWECiphertext ct2);
LWECiphertext EvalNOT(BinFHEContext ctx, LWECiphertext ct);
LWECiphertext EvalNAND(BinFHEContext ctx, LWECiphertext ct1, LWECiphertext ct2);
LWECiphertext EvalNOR(BinFHEContext ctx, LWECiphertext ct1, LWECiphertext ct2);
LWECiphertext EvalXNOR(BinFHEContext ctx, LWECiphertext ct1, LWECiphertext ct2);
*/
import "C"
import (
	"runtime"
	"unsafe"
)

// SecurityLevel defines the security parameter set.
type SecurityLevel int

const (
	// STD128 provides 128-bit security (recommended)
	STD128 SecurityLevel = iota
	// STD192 provides 192-bit security
	STD192
	// STD256 provides 256-bit security
	STD256
)

// Method defines the bootstrapping method.
type Method int

const (
	// GINX is the default bootstrapping method (fastest)
	GINX Method = iota
	// AP is the original FHEW method
	AP
	// LMKCDEY is optimized for large precision
	LMKCDEY
)

// Context holds the TFHE cryptographic context.
type Context struct {
	ptr C.BinFHEContext
}

// SecretKey holds a TFHE secret key.
type SecretKey struct {
	ptr C.LWESecretKey
}

// Ciphertext holds a TFHE ciphertext (encrypted bit).
type Ciphertext struct {
	ptr C.LWECiphertext
}

// NewContext creates a new TFHE context with the given security level.
func NewContext(security SecurityLevel) *Context {
	ctx := &Context{
		ptr: C.NewBinFHEContext(),
	}
	C.GenerateBinFHEContext(ctx.ptr, C.int(security), C.int(GINX))
	runtime.SetFinalizer(ctx, (*Context).Close)
	return ctx
}

// NewContextWithMethod creates a context with a specific bootstrapping method.
func NewContextWithMethod(security SecurityLevel, method Method) *Context {
	ctx := &Context{
		ptr: C.NewBinFHEContext(),
	}
	C.GenerateBinFHEContext(ctx.ptr, C.int(security), C.int(method))
	runtime.SetFinalizer(ctx, (*Context).Close)
	return ctx
}

// Close releases the context resources.
func (c *Context) Close() {
	if c.ptr != nil {
		C.FreeBinFHEContext(c.ptr)
		c.ptr = nil
	}
}

// KeyGen generates a new secret key.
func (c *Context) KeyGen() *SecretKey {
	sk := &SecretKey{
		ptr: C.KeyGen(c.ptr),
	}
	runtime.SetFinalizer(sk, (*SecretKey).free)
	return sk
}

func (sk *SecretKey) free() {
	if sk.ptr != nil {
		C.FreeLWESecretKey(sk.ptr)
		sk.ptr = nil
	}
}

// BootstrapKeyGen generates the bootstrapping key (required for gate operations).
func (c *Context) BootstrapKeyGen(sk *SecretKey) {
	C.BTKeyGen(c.ptr, sk.ptr)
}

// Encrypt encrypts a boolean value.
func (c *Context) Encrypt(sk *SecretKey, value bool) *Ciphertext {
	var v C.int
	if value {
		v = 1
	}
	ct := &Ciphertext{
		ptr: C.Encrypt(c.ptr, sk.ptr, v),
	}
	runtime.SetFinalizer(ct, (*Ciphertext).free)
	return ct
}

func (ct *Ciphertext) free() {
	if ct.ptr != nil {
		C.FreeLWECiphertext(ct.ptr)
		ct.ptr = nil
	}
}

// Decrypt decrypts a ciphertext to a boolean.
func (c *Context) Decrypt(sk *SecretKey, ct *Ciphertext) bool {
	result := C.Decrypt(c.ptr, sk.ptr, ct.ptr)
	return result != 0
}

// AND performs homomorphic AND gate.
func (c *Context) AND(a, b *Ciphertext) *Ciphertext {
	ct := &Ciphertext{
		ptr: C.EvalAND(c.ptr, a.ptr, b.ptr),
	}
	runtime.SetFinalizer(ct, (*Ciphertext).free)
	return ct
}

// OR performs homomorphic OR gate.
func (c *Context) OR(a, b *Ciphertext) *Ciphertext {
	ct := &Ciphertext{
		ptr: C.EvalOR(c.ptr, a.ptr, b.ptr),
	}
	runtime.SetFinalizer(ct, (*Ciphertext).free)
	return ct
}

// XOR performs homomorphic XOR gate.
func (c *Context) XOR(a, b *Ciphertext) *Ciphertext {
	ct := &Ciphertext{
		ptr: C.EvalXOR(c.ptr, a.ptr, b.ptr),
	}
	runtime.SetFinalizer(ct, (*Ciphertext).free)
	return ct
}

// NOT performs homomorphic NOT gate.
func (c *Context) NOT(a *Ciphertext) *Ciphertext {
	ct := &Ciphertext{
		ptr: C.EvalNOT(c.ptr, a.ptr),
	}
	runtime.SetFinalizer(ct, (*Ciphertext).free)
	return ct
}

// NAND performs homomorphic NAND gate.
func (c *Context) NAND(a, b *Ciphertext) *Ciphertext {
	ct := &Ciphertext{
		ptr: C.EvalNAND(c.ptr, a.ptr, b.ptr),
	}
	runtime.SetFinalizer(ct, (*Ciphertext).free)
	return ct
}

// NOR performs homomorphic NOR gate.
func (c *Context) NOR(a, b *Ciphertext) *Ciphertext {
	ct := &Ciphertext{
		ptr: C.EvalNOR(c.ptr, a.ptr, b.ptr),
	}
	runtime.SetFinalizer(ct, (*Ciphertext).free)
	return ct
}

// XNOR performs homomorphic XNOR gate.
func (c *Context) XNOR(a, b *Ciphertext) *Ciphertext {
	ct := &Ciphertext{
		ptr: C.EvalXNOR(c.ptr, a.ptr, b.ptr),
	}
	runtime.SetFinalizer(ct, (*Ciphertext).free)
	return ct
}

// Pointer returns the raw C pointer (for advanced usage).
func (c *Context) Pointer() unsafe.Pointer {
	return unsafe.Pointer(c.ptr)
}
