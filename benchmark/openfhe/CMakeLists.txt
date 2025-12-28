cmake_minimum_required(VERSION 3.16)
project(fhe-benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenFHE installed location
set(OPENFHE_INSTALL_DIR "${CMAKE_SOURCE_DIR}/../../install")

# Find OpenFHE using cmake config
list(APPEND CMAKE_PREFIX_PATH "${OPENFHE_INSTALL_DIR}/lib/OpenFHE")
find_package(OpenFHE REQUIRED)

# Include directories
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)

# Third-party cereal
include_directories(${CMAKE_SOURCE_DIR}/../../third-party/cereal/include)

# Link directories
link_directories(${OpenFHE_LIBDIR})

# OpenMP (optional)
if(OPENMP_INCLUDES)
    include_directories(${OPENMP_INCLUDES})
    link_directories(${OPENMP_LIBRARIES})
endif()

# CKKS benchmark (primary - for comparison with Go)
add_executable(ckks_bench ckks_bench.cpp)
target_link_libraries(ckks_bench ${OpenFHE_LIBRARIES})

# BinFHE benchmark
add_executable(binfhe_bench binfhe_bench.cpp)
target_link_libraries(binfhe_bench ${OpenFHE_LIBRARIES})

# PKE benchmark (CKKS, BGV, BFV)
add_executable(pke_bench pke_bench.cpp)
target_link_libraries(pke_bench ${OpenFHE_LIBRARIES})
