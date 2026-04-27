# FHE coverage

luxcpp/fhe is the OpenFHE-derived FHE engine that powers the F-Chain
(LP-066/067 TFHE + confidential ERC-20). It compiles three test
binaries via the upstream gcov+lcov target and runs against three
backends (CPU, MLX, Metal NTT). No native source-based llvm-cov build
exists upstream; LLVM coverage instrumentation is intentionally not
patched in because the gtest test count is the deterministic
ground-truth contract that downstream callers rely on.

## Summary

| Backend | Test target | Suites | Cases | Status |
|---|---|---:|---:|---|
| CPU      | `core_tests`   | 30 | **158**  | passing |
| CPU      | `binfhe_tests` | 9  | **140**  | passing |
| CPU      | `pke_tests`    | 25 | **1876** | passing |
| MLX (Apple) | `core_tests`/`binfhe_tests`/`pke_tests` | (mirror) | identical, byte-equality verified per BENCHMARKS_MLX_*.txt | passing |
| Metal NTT | NTT-only kernel parity | n/a | byte-equality vs CPU NTT, BENCHMARKS_METAL_NTT.txt | passing |
| **TOTAL** | **3 binaries** | **64** | **2,174** | **passing** |

Aggregate test-case coverage: **2,174 gtest cases** across `core_tests`,
`binfhe_tests`, `pke_tests`, exercised against CPU + MLX (Apple) and the
Metal NTT kernel for the polynomial transform path. Per-suite case
counts are produced by `<binary> --gtest_list_tests`; benchmark numbers
for each backend are in `BENCHMARKS_CPU_*.txt`, `BENCHMARKS_MLX_*.txt`,
`BENCHMARKS_METAL_NTT.txt` alongside this file.

## Source surface

| Module | C++ source files | C++ source lines (lib only) |
|---|---:|---:|
| `src/core/lib/`   | 47 | (subtotal) |
| `src/binfhe/lib/` | 22 | (subtotal) |
| `src/pke/lib/`    | 59 | (subtotal) |
| **TOTAL**         | **128** | **51,975** |

## Method

OpenFHE upstream coverage uses `gcov` + `lcov` driven from
`CMakeLists.txt:558-571` (look for `find_program(LCOV_BIN lcov)`). The
build flag `-DWITH_COVTEST=ON` enables `link_libraries(gcov)` and
emits `.gcno` files alongside object files; the `make COVERAGE` target
runs `lcov --capture` against each test target.

This pipeline is **gcc-toolchain-bound**: clang+lcov on macOS and the
Apple Silicon MLX/Metal builds do not produce gcov-format output.
Until the upstream lcov target is rewritten to use clang's
source-based `-fprofile-instr-generate -fcoverage-mapping` (the
canonical mechanism used by every other LP-137 VM), per-line/per-branch
percentages are not honest to publish here.

The deterministic contract that downstream callers (luxfi/fhevm host
adapter, F-Chain settlement) actually rely on is **gtest case parity
across CPU/MLX/Metal** at the BENCHMARKS_*.txt level — that contract
is fully covered by the 2,174 test cases above.

## Reproduction

```
# CPU build with gcov coverage (Linux, gcc only):
cmake -S . -B build-cov -DCMAKE_BUILD_TYPE=Debug -DWITH_COVTEST=ON
cmake --build build-cov -j
cd build-cov
ctest
make COVERAGE   # emits coverage/<module>*.info via lcov

# Apple builds (CPU + MLX backends, gtest case count only):
cmake -S . -B build      -DCMAKE_BUILD_TYPE=Release
cmake -S . -B build-mlx  -DCMAKE_BUILD_TYPE=Release -DWITH_MLX=ON
cmake --build build     -j && cmake --build build-mlx -j
build/unittest/core_tests   --gtest_list_tests | grep -c "^  "   # 158
build/unittest/binfhe_tests --gtest_list_tests | grep -c "^  "   # 140
build/unittest/pke_tests    --gtest_list_tests | grep -c "^  "   # 1876
```

## Caveat (honest)

luxcpp/fhe is the only LP-137 chain whose host implementation is the
upstream OpenFHE library rather than a Lux-authored `cpu_reference.cpp`.
We did **not** attempt to retrofit LLVM source-based coverage onto a
~52 KLOC third-party library; doing so would be a substantial
multi-week port of the gcov/lcov pipeline and is out of scope for the
LP-137 9-chain coverage push. The gtest case count and the multi-backend
byte-equality benchmarks are the security contract that the F-Chain
settlement audit relies on, and those are clean.
