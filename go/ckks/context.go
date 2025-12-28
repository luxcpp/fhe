// Package ckks provides Go bindings for the CKKS FHE scheme via OpenFHE.
//
// CKKS enables approximate arithmetic on encrypted real numbers,
// making it suitable for ML inference and scientific computation.
//
// Build with CGO:
//
//	go build -tags=openfhe
//
// Without the tag, this package provides stub implementations that
// defer to luxfi/lattice for pure Go execution.
package ckks

/*
#cgo CXXFLAGS: -std=c++17 -O3 -I${SRCDIR}/../../install/include/openfhe -I${SRCDIR}/../../install/include/openfhe/pke -I${SRCDIR}/../../install/include/openfhe/core -I${SRCDIR}/../../install/include/openfhe/binfhe
#cgo LDFLAGS: -L${SRCDIR}/../../install/lib -Wl,-rpath,${SRCDIR}/../../install/lib -lOPENFHEpke -lOPENFHEcore -lOPENFHEbinfhe

#include <stdlib.h>

// Forward declarations - implemented in bridge.cpp
typedef void* CKKSContextPtr;
typedef void* CKKSKeyPairPtr;
typedef void* CKKSCiphertextPtr;
typedef void* CKKSPlaintextPtr;

// Context management
CKKSContextPtr ckks_context_new(int log_n, int log_q, double scale);
void ckks_context_free(CKKSContextPtr ctx);

// Key generation
CKKSKeyPairPtr ckks_keygen(CKKSContextPtr ctx);
void ckks_keypair_free(CKKSKeyPairPtr kp);

// Encryption/Decryption
CKKSCiphertextPtr ckks_encrypt(CKKSContextPtr ctx, CKKSKeyPairPtr kp, double* values, int len);
double* ckks_decrypt(CKKSContextPtr ctx, CKKSKeyPairPtr kp, CKKSCiphertextPtr ct, int* out_len);
void ckks_ciphertext_free(CKKSCiphertextPtr ct);

// Homomorphic operations
CKKSCiphertextPtr ckks_add(CKKSContextPtr ctx, CKKSCiphertextPtr a, CKKSCiphertextPtr b);
CKKSCiphertextPtr ckks_sub(CKKSContextPtr ctx, CKKSCiphertextPtr a, CKKSCiphertextPtr b);
CKKSCiphertextPtr ckks_mult(CKKSContextPtr ctx, CKKSCiphertextPtr a, CKKSCiphertextPtr b);
CKKSCiphertextPtr ckks_rotate(CKKSContextPtr ctx, CKKSKeyPairPtr kp, CKKSCiphertextPtr ct, int steps);

// Bootstrapping
CKKSCiphertextPtr ckks_bootstrap(CKKSContextPtr ctx, CKKSKeyPairPtr kp, CKKSCiphertextPtr ct);

// Serialization
unsigned char* ckks_serialize_ciphertext(CKKSCiphertextPtr ct, int* out_len);
CKKSCiphertextPtr ckks_deserialize_ciphertext(CKKSContextPtr ctx, unsigned char* data, int len);
*/
import "C"

import (
	"errors"
	"runtime"
	"unsafe"
)

var (
	ErrNilContext    = errors.New("ckks: nil context")
	ErrNilKeyPair    = errors.New("ckks: nil keypair")
	ErrNilCiphertext = errors.New("ckks: nil ciphertext")
	ErrEncryption    = errors.New("ckks: encryption failed")
	ErrDecryption    = errors.New("ckks: decryption failed")
)

// SecurityLevel represents CKKS security parameters
type SecurityLevel int

const (
	// HES (Homomorphic Encryption Standard) security levels
	Security128 SecurityLevel = 128
	Security192 SecurityLevel = 192
	Security256 SecurityLevel = 256
)

// Parameters holds CKKS scheme parameters
type Parameters struct {
	LogN         int     // Ring dimension (log2)
	LogQ         int     // Ciphertext modulus (log2)
	Scale        float64 // Encoding scale
	SecurityBits int     // Security level in bits
}

// DefaultParameters returns production-ready CKKS parameters
func DefaultParameters() Parameters {
	return Parameters{
		LogN:         14,     // N = 16384
		LogQ:         438,    // ~438-bit modulus
		Scale:        1 << 40, // 2^40 scale
		SecurityBits: 128,
	}
}

// Context holds CKKS cryptographic context (wraps OpenFHE CryptoContext)
type Context struct {
	ptr    C.CKKSContextPtr
	params Parameters
}

// NewContext creates a CKKS context with given parameters
func NewContext(params Parameters) (*Context, error) {
	ptr := C.ckks_context_new(
		C.int(params.LogN),
		C.int(params.LogQ),
		C.double(params.Scale),
	)
	if ptr == nil {
		return nil, errors.New("ckks: failed to create context")
	}

	ctx := &Context{ptr: ptr, params: params}
	runtime.SetFinalizer(ctx, (*Context).Free)
	return ctx, nil
}

// Free releases the context resources
func (c *Context) Free() {
	if c.ptr != nil {
		C.ckks_context_free(c.ptr)
		c.ptr = nil
	}
}

// Parameters returns the context parameters
func (c *Context) Parameters() Parameters {
	return c.params
}

// KeyPair holds CKKS public and secret keys
type KeyPair struct {
	ptr C.CKKSKeyPairPtr
	ctx *Context
}

// KeyGen generates a new key pair
func (c *Context) KeyGen() (*KeyPair, error) {
	if c.ptr == nil {
		return nil, ErrNilContext
	}

	ptr := C.ckks_keygen(c.ptr)
	if ptr == nil {
		return nil, errors.New("ckks: keygen failed")
	}

	kp := &KeyPair{ptr: ptr, ctx: c}
	runtime.SetFinalizer(kp, (*KeyPair).Free)
	return kp, nil
}

// Free releases the key pair resources
func (kp *KeyPair) Free() {
	if kp.ptr != nil {
		C.ckks_keypair_free(kp.ptr)
		kp.ptr = nil
	}
}

// Ciphertext holds encrypted data
type Ciphertext struct {
	ptr C.CKKSCiphertextPtr
	ctx *Context
}

// Free releases the ciphertext resources
func (ct *Ciphertext) Free() {
	if ct.ptr != nil {
		C.ckks_ciphertext_free(ct.ptr)
		ct.ptr = nil
	}
}

// Serialize converts ciphertext to bytes for storage/transmission
func (ct *Ciphertext) Serialize() ([]byte, error) {
	if ct.ptr == nil {
		return nil, ErrNilCiphertext
	}

	var outLen C.int
	data := C.ckks_serialize_ciphertext(ct.ptr, &outLen)
	if data == nil {
		return nil, errors.New("ckks: serialization failed")
	}
	defer C.free(unsafe.Pointer(data))

	return C.GoBytes(unsafe.Pointer(data), outLen), nil
}

// Deserialize creates a ciphertext from bytes
func (c *Context) Deserialize(data []byte) (*Ciphertext, error) {
	if c.ptr == nil {
		return nil, ErrNilContext
	}

	ptr := C.ckks_deserialize_ciphertext(
		c.ptr,
		(*C.uchar)(unsafe.Pointer(&data[0])),
		C.int(len(data)),
	)
	if ptr == nil {
		return nil, errors.New("ckks: deserialization failed")
	}

	ct := &Ciphertext{ptr: ptr, ctx: c}
	runtime.SetFinalizer(ct, (*Ciphertext).Free)
	return ct, nil
}

// Encrypt encrypts a vector of float64 values
func (c *Context) Encrypt(kp *KeyPair, values []float64) (*Ciphertext, error) {
	if c.ptr == nil {
		return nil, ErrNilContext
	}
	if kp.ptr == nil {
		return nil, ErrNilKeyPair
	}
	if len(values) == 0 {
		return nil, errors.New("ckks: empty input")
	}

	ptr := C.ckks_encrypt(
		c.ptr,
		kp.ptr,
		(*C.double)(unsafe.Pointer(&values[0])),
		C.int(len(values)),
	)
	if ptr == nil {
		return nil, ErrEncryption
	}

	ct := &Ciphertext{ptr: ptr, ctx: c}
	runtime.SetFinalizer(ct, (*Ciphertext).Free)
	return ct, nil
}

// Decrypt decrypts a ciphertext to float64 values
func (c *Context) Decrypt(kp *KeyPair, ct *Ciphertext) ([]float64, error) {
	if c.ptr == nil {
		return nil, ErrNilContext
	}
	if kp.ptr == nil {
		return nil, ErrNilKeyPair
	}
	if ct.ptr == nil {
		return nil, ErrNilCiphertext
	}

	var outLen C.int
	data := C.ckks_decrypt(c.ptr, kp.ptr, ct.ptr, &outLen)
	if data == nil {
		return nil, ErrDecryption
	}
	defer C.free(unsafe.Pointer(data))

	// Copy C array to Go slice
	length := int(outLen)
	result := make([]float64, length)
	cSlice := (*[1 << 30]C.double)(unsafe.Pointer(data))[:length:length]
	for i, v := range cSlice {
		result[i] = float64(v)
	}

	return result, nil
}

// Add performs homomorphic addition: result = a + b
func (c *Context) Add(a, b *Ciphertext) (*Ciphertext, error) {
	if c.ptr == nil {
		return nil, ErrNilContext
	}
	if a.ptr == nil || b.ptr == nil {
		return nil, ErrNilCiphertext
	}

	ptr := C.ckks_add(c.ptr, a.ptr, b.ptr)
	if ptr == nil {
		return nil, errors.New("ckks: add failed")
	}

	ct := &Ciphertext{ptr: ptr, ctx: c}
	runtime.SetFinalizer(ct, (*Ciphertext).Free)
	return ct, nil
}

// Sub performs homomorphic subtraction: result = a - b
func (c *Context) Sub(a, b *Ciphertext) (*Ciphertext, error) {
	if c.ptr == nil {
		return nil, ErrNilContext
	}
	if a.ptr == nil || b.ptr == nil {
		return nil, ErrNilCiphertext
	}

	ptr := C.ckks_sub(c.ptr, a.ptr, b.ptr)
	if ptr == nil {
		return nil, errors.New("ckks: sub failed")
	}

	ct := &Ciphertext{ptr: ptr, ctx: c}
	runtime.SetFinalizer(ct, (*Ciphertext).Free)
	return ct, nil
}

// Mult performs homomorphic multiplication: result = a * b
func (c *Context) Mult(a, b *Ciphertext) (*Ciphertext, error) {
	if c.ptr == nil {
		return nil, ErrNilContext
	}
	if a.ptr == nil || b.ptr == nil {
		return nil, ErrNilCiphertext
	}

	ptr := C.ckks_mult(c.ptr, a.ptr, b.ptr)
	if ptr == nil {
		return nil, errors.New("ckks: mult failed")
	}

	ct := &Ciphertext{ptr: ptr, ctx: c}
	runtime.SetFinalizer(ct, (*Ciphertext).Free)
	return ct, nil
}

// Rotate performs vector rotation by steps positions
func (c *Context) Rotate(kp *KeyPair, ct *Ciphertext, steps int) (*Ciphertext, error) {
	if c.ptr == nil {
		return nil, ErrNilContext
	}
	if kp.ptr == nil {
		return nil, ErrNilKeyPair
	}
	if ct.ptr == nil {
		return nil, ErrNilCiphertext
	}

	ptr := C.ckks_rotate(c.ptr, kp.ptr, ct.ptr, C.int(steps))
	if ptr == nil {
		return nil, errors.New("ckks: rotate failed")
	}

	result := &Ciphertext{ptr: ptr, ctx: c}
	runtime.SetFinalizer(result, (*Ciphertext).Free)
	return result, nil
}

// Bootstrap refreshes ciphertext noise (expensive operation)
func (c *Context) Bootstrap(kp *KeyPair, ct *Ciphertext) (*Ciphertext, error) {
	if c.ptr == nil {
		return nil, ErrNilContext
	}
	if kp.ptr == nil {
		return nil, ErrNilKeyPair
	}
	if ct.ptr == nil {
		return nil, ErrNilCiphertext
	}

	ptr := C.ckks_bootstrap(c.ptr, kp.ptr, ct.ptr)
	if ptr == nil {
		return nil, errors.New("ckks: bootstrap failed")
	}

	result := &Ciphertext{ptr: ptr, ctx: c}
	runtime.SetFinalizer(result, (*Ciphertext).Free)
	return result, nil
}
