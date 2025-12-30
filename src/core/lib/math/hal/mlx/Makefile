# Lux FHE MLX/Metal Benchmark Makefile

CXX = clang++
CXXFLAGS = -std=c++17 -O3 -DWITH_MLX -Wall -Wextra

# Paths (from src/core/lib/math/hal/mlx/)
FHE_ROOT = ../../../../../..
CORE_INCLUDE = $(FHE_ROOT)/src/core/include
CORE_LIB = $(FHE_ROOT)/src/core/lib
MLX_ROOT = $(FHE_ROOT)/.venv/lib/python3.12/site-packages/mlx

INCLUDES = -I$(CORE_INCLUDE) -I$(CORE_LIB) -I$(MLX_ROOT)/include
LDFLAGS = -L$(MLX_ROOT)/lib -lmlx -framework Metal -framework Foundation
RPATH = -Wl,-rpath,$(MLX_ROOT)/lib

.PHONY: all clean bench

all: bench

bench: standalone_bench.cpp ntt.h ntt_fourstep.h gpu_primes.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) standalone_bench.cpp -o bench $(LDFLAGS) $(RPATH)

run: bench
	./bench

clean:
	rm -f bench

# Quick syntax check (no linking)
check:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -fsyntax-only standalone_bench.cpp

# Check all headers compile
headers:
	@for h in *.h; do \
		echo -n "$$h: "; \
		$(CXX) $(CXXFLAGS) $(INCLUDES) -fsyntax-only "$$h" 2>&1 && echo "OK" || echo "FAIL"; \
	done
