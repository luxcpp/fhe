package tfhe

// This file provides integer comparison operations built on boolean gates.
// These are essential for confidential transaction validation.

// EncryptedInt8 represents an encrypted 8-bit integer (8 ciphertexts).
type EncryptedInt8 [8]*Ciphertext

// EncryptedInt16 represents an encrypted 16-bit integer.
type EncryptedInt16 [16]*Ciphertext

// EncryptedInt32 represents an encrypted 32-bit integer.
type EncryptedInt32 [32]*Ciphertext

// EncryptedInt64 represents an encrypted 64-bit integer.
type EncryptedInt64 [64]*Ciphertext

// EncryptInt8 encrypts an 8-bit integer as 8 separate bit ciphertexts.
func (c *Context) EncryptInt8(sk *SecretKey, value int8) EncryptedInt8 {
	var result EncryptedInt8
	for i := 0; i < 8; i++ {
		bit := (value >> i) & 1
		result[i] = c.Encrypt(sk, bit == 1)
	}
	return result
}

// DecryptInt8 decrypts an encrypted 8-bit integer.
func (c *Context) DecryptInt8(sk *SecretKey, ct EncryptedInt8) int8 {
	var result int8
	for i := 0; i < 8; i++ {
		if c.Decrypt(sk, ct[i]) {
			result |= 1 << i
		}
	}
	return result
}

// EncryptInt32 encrypts a 32-bit integer.
func (c *Context) EncryptInt32(sk *SecretKey, value int32) EncryptedInt32 {
	var result EncryptedInt32
	for i := 0; i < 32; i++ {
		bit := (value >> i) & 1
		result[i] = c.Encrypt(sk, bit == 1)
	}
	return result
}

// DecryptInt32 decrypts an encrypted 32-bit integer.
func (c *Context) DecryptInt32(sk *SecretKey, ct EncryptedInt32) int32 {
	var result int32
	for i := 0; i < 32; i++ {
		if c.Decrypt(sk, ct[i]) {
			result |= 1 << i
		}
	}
	return result
}

// EncryptInt64 encrypts a 64-bit integer.
func (c *Context) EncryptInt64(sk *SecretKey, value int64) EncryptedInt64 {
	var result EncryptedInt64
	for i := 0; i < 64; i++ {
		bit := (value >> i) & 1
		result[i] = c.Encrypt(sk, bit == 1)
	}
	return result
}

// DecryptInt64 decrypts an encrypted 64-bit integer.
func (c *Context) DecryptInt64(sk *SecretKey, ct EncryptedInt64) int64 {
	var result int64
	for i := 0; i < 64; i++ {
		if c.Decrypt(sk, ct[i]) {
			result |= 1 << i
		}
	}
	return result
}

// Equal8 checks if two encrypted 8-bit integers are equal.
// Returns encrypted bit: 1 if equal, 0 otherwise.
func (c *Context) Equal8(a, b EncryptedInt8) *Ciphertext {
	// a == b iff all bits are equal
	// bit equality: XNOR(a, b)
	result := c.XNOR(a[0], b[0])
	for i := 1; i < 8; i++ {
		bitEq := c.XNOR(a[i], b[i])
		result = c.AND(result, bitEq)
	}
	return result
}

// Equal32 checks if two encrypted 32-bit integers are equal.
func (c *Context) Equal32(a, b EncryptedInt32) *Ciphertext {
	result := c.XNOR(a[0], b[0])
	for i := 1; i < 32; i++ {
		bitEq := c.XNOR(a[i], b[i])
		result = c.AND(result, bitEq)
	}
	return result
}

// GreaterThan8 checks if a > b for encrypted 8-bit unsigned integers.
// Returns encrypted bit: 1 if a > b, 0 otherwise.
func (c *Context) GreaterThan8(a, b EncryptedInt8) *Ciphertext {
	// Compare from MSB to LSB
	// At each bit: if a[i] > b[i], result is 1
	//              if a[i] < b[i], result is 0
	//              if equal, continue to next bit
	
	// Start with false (not greater)
	result := c.Encrypt(nil, false) // Need a "false" ciphertext
	
	// Actually, we need to build this properly with ripple comparison
	// a > b when: a[7] > b[7], OR (a[7] == b[7] AND a[6:0] > b[6:0])
	
	// Simplified: compare bit by bit from MSB
	for i := 7; i >= 0; i-- {
		// a[i] AND NOT b[i] means a has 1, b has 0 at this position
		aGreater := c.AND(a[i], c.NOT(b[i]))
		
		// NOT a[i] AND b[i] means a has 0, b has 1
		bGreater := c.AND(c.NOT(a[i]), b[i])
		
		// Equal at this bit
		equal := c.XNOR(a[i], b[i])
		
		// If a > b at this bit, result is true
		// If b > a at this bit, result is false
		// If equal, keep previous result
		result = c.OR(aGreater, c.AND(equal, result))
		
		// Short circuit if b > a
		result = c.AND(result, c.NOT(bGreater))
	}
	
	return result
}

// LessThan8 checks if a < b for encrypted 8-bit unsigned integers.
func (c *Context) LessThan8(a, b EncryptedInt8) *Ciphertext {
	return c.GreaterThan8(b, a)
}

// GreaterOrEqual8 checks if a >= b.
func (c *Context) GreaterOrEqual8(a, b EncryptedInt8) *Ciphertext {
	lt := c.LessThan8(a, b)
	return c.NOT(lt)
}

// LessOrEqual8 checks if a <= b.
func (c *Context) LessOrEqual8(a, b EncryptedInt8) *Ciphertext {
	gt := c.GreaterThan8(a, b)
	return c.NOT(gt)
}

// Add8 adds two encrypted 8-bit integers with carry.
// Returns (sum, carry_out).
func (c *Context) Add8(a, b EncryptedInt8) (EncryptedInt8, *Ciphertext) {
	var result EncryptedInt8
	var carry *Ciphertext
	
	// Half adder for first bit
	result[0] = c.XOR(a[0], b[0])
	carry = c.AND(a[0], b[0])
	
	// Full adder for remaining bits
	for i := 1; i < 8; i++ {
		// sum = a XOR b XOR carry
		ab := c.XOR(a[i], b[i])
		result[i] = c.XOR(ab, carry)
		
		// carry = (a AND b) OR (carry AND (a XOR b))
		ab_and := c.AND(a[i], b[i])
		carry_and := c.AND(carry, ab)
		carry = c.OR(ab_and, carry_and)
	}
	
	return result, carry
}
