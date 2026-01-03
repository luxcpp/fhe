#!/bin/bash
# Rebrand OpenFHE to Lux FHE
# Run from luxcpp/fhe directory

set -e

echo "=== Lux FHE Rebranding Script ==="
echo ""

# Find all source files
FILES=$(find src benchmark -type f \( -name "*.h" -o -name "*.hpp" -o -name "*.cpp" -o -name "*.mm" \) 2>/dev/null)

echo "Found $(echo "$FILES" | wc -l | tr -d ' ') source files"

# 1. Namespace changes
echo ""
echo "[1/6] Changing namespace lbcrypto -> lux::fhe..."
for f in $FILES; do
    sed -i '' 's/namespace lbcrypto/namespace lux::fhe/g' "$f" 2>/dev/null || true
done

echo "[2/6] Changing lbcrypto:: -> lux::fhe::..."
for f in $FILES; do
    sed -i '' 's/lbcrypto::/lux::fhe::/g' "$f" 2>/dev/null || true
done

echo "[3/6] Changing using namespace..."
for f in $FILES; do
    sed -i '' 's/using namespace lux::fhe::fhe/using namespace lux::fhe/g' "$f" 2>/dev/null || true
done

# 2. Macro changes
echo "[4/6] Changing OPENFHE_ macros -> LUX_FHE_..."
for f in $FILES; do
    sed -i '' 's/OPENFHE_/LUX_FHE_/g' "$f" 2>/dev/null || true
done

# 3. GPU namespace (nested under lbcrypto -> now under lux)
echo "[5/6] Changing lux::fhe::gpu:: -> lux::gpu::..."
for f in $FILES; do
    sed -i '' 's/lux::fhe::gpu::/lux::gpu::/g' "$f" 2>/dev/null || true
    # Also handle the nested namespace declaration
    sed -i '' 's/namespace lux::fhe {[[:space:]]*namespace gpu/namespace lux::gpu/g' "$f" 2>/dev/null || true
done

# 4. Include paths
echo "[6/6] Updating include paths..."
for f in $FILES; do
    sed -i '' 's|#include "openfhe|#include "lux/fhe|g' "$f" 2>/dev/null || true
    sed -i '' 's|#include <openfhe|#include <lux/fhe|g' "$f" 2>/dev/null || true
done

echo ""
echo "=== Done ==="
echo ""
echo "Next steps:"
echo "  1. Review changes: git diff"
echo "  2. Fix any edge cases manually"
echo "  3. Update CMakeLists.txt"
echo "  4. Rebuild: cmake --build build"
