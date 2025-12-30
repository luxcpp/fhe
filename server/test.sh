#!/bin/bash
# Test script for Lux FHE HTTP Server
# Usage: ./test.sh [port]

PORT=${1:-8080}
BASE_URL="http://localhost:$PORT"

echo "=== Lux FHE Server Test Suite ==="
echo "Target: $BASE_URL"
echo ""

# Health check
echo "1. Health check..."
curl -s "$BASE_URL/health" | jq .
echo ""

# Create context
echo "2. Creating FHE context..."
CTX_RESPONSE=$(curl -s -X POST "$BASE_URL/v1/context/create" \
    -H "Content-Type: application/json" \
    -d '{"security": "STD128_LMKCDEY", "method": "LMKCDEY"}')
echo "$CTX_RESPONSE" | jq .
CTX_ID=$(echo "$CTX_RESPONSE" | jq -r '.context_id')
echo "Context ID: $CTX_ID"
echo ""

# Generate keys
echo "3. Generating keys..."
curl -s -X POST "$BASE_URL/v1/keys/generate" \
    -H "Content-Type: application/json" \
    -d "{\"context_id\": \"$CTX_ID\"}" | jq .
echo ""

# Encrypt true
echo "4. Encrypting true..."
CT1_RESPONSE=$(curl -s -X POST "$BASE_URL/v1/encrypt" \
    -H "Content-Type: application/json" \
    -d "{\"context_id\": \"$CTX_ID\", \"value\": true}")
echo "$CT1_RESPONSE" | jq .
CT1_ID=$(echo "$CT1_RESPONSE" | jq -r '.ciphertext_id')
echo ""

# Encrypt false
echo "5. Encrypting false..."
CT2_RESPONSE=$(curl -s -X POST "$BASE_URL/v1/encrypt" \
    -H "Content-Type: application/json" \
    -d "{\"context_id\": \"$CTX_ID\", \"value\": false}")
echo "$CT2_RESPONSE" | jq .
CT2_ID=$(echo "$CT2_RESPONSE" | jq -r '.ciphertext_id')
echo ""

# AND gate: true AND false = false
echo "6. Evaluating AND gate (true AND false)..."
AND_RESPONSE=$(curl -s -X POST "$BASE_URL/v1/eval/and" \
    -H "Content-Type: application/json" \
    -d "{\"ct1_id\": \"$CT1_ID\", \"ct2_id\": \"$CT2_ID\"}")
echo "$AND_RESPONSE" | jq .
AND_ID=$(echo "$AND_RESPONSE" | jq -r '.result_id')
echo ""

# Decrypt AND result
echo "7. Decrypting AND result..."
curl -s -X POST "$BASE_URL/v1/decrypt" \
    -H "Content-Type: application/json" \
    -d "{\"ciphertext_id\": \"$AND_ID\"}" | jq .
echo ""

# OR gate: true OR false = true
echo "8. Evaluating OR gate (true OR false)..."
OR_RESPONSE=$(curl -s -X POST "$BASE_URL/v1/eval/or" \
    -H "Content-Type: application/json" \
    -d "{\"ct1_id\": \"$CT1_ID\", \"ct2_id\": \"$CT2_ID\"}")
echo "$OR_RESPONSE" | jq .
OR_ID=$(echo "$OR_RESPONSE" | jq -r '.result_id')
echo ""

# Decrypt OR result
echo "9. Decrypting OR result..."
curl -s -X POST "$BASE_URL/v1/decrypt" \
    -H "Content-Type: application/json" \
    -d "{\"ciphertext_id\": \"$OR_ID\"}" | jq .
echo ""

# NOT gate: NOT true = false
echo "10. Evaluating NOT gate (NOT true)..."
NOT_RESPONSE=$(curl -s -X POST "$BASE_URL/v1/eval/not" \
    -H "Content-Type: application/json" \
    -d "{\"ciphertext_id\": \"$CT1_ID\"}")
echo "$NOT_RESPONSE" | jq .
NOT_ID=$(echo "$NOT_RESPONSE" | jq -r '.result_id')
echo ""

# Decrypt NOT result
echo "11. Decrypting NOT result..."
curl -s -X POST "$BASE_URL/v1/decrypt" \
    -H "Content-Type: application/json" \
    -d "{\"ciphertext_id\": \"$NOT_ID\"}" | jq .
echo ""

# Stats
echo "12. Server stats..."
curl -s "$BASE_URL/v1/stats" | jq .
echo ""

echo "=== Test Complete ==="
