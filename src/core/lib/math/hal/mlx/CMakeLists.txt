# MLX GPU Backend CMakeLists.txt
#
# GPU acceleration using Apple's MLX framework and native Metal kernels
# Enable with: cmake -DWITH_MLX=ON ..
#
# Optimization levels:
# 1. MLX operations (easy, portable)
# 2. Native Metal kernels (10x+ speedup for NTT)

if(WITH_MLX)
    message(STATUS "MLX GPU Backend: Enabled")

    # Check for Apple Silicon
    if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        message(WARNING "MLX is only supported on Apple Silicon (arm64). Disabling MLX backend.")
        set(WITH_MLX OFF PARENT_SCOPE)
        return()
    endif()

    # Find Metal framework (required for native kernels)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    message(STATUS "Found Metal framework: ${METAL_FRAMEWORK}")

    # Find MLX
    find_package(MLX CONFIG QUIET)

    if(NOT MLX_FOUND)
        # Try to find MLX in common locations
        find_path(MLX_INCLUDE_DIR
            NAMES mlx/mlx.h
            PATHS
                ${CMAKE_SOURCE_DIR}/.venv/lib/python3.12/site-packages/mlx/include
                ${CMAKE_SOURCE_DIR}/.venv/lib/python3.11/site-packages/mlx/include
                /opt/homebrew/include
                /usr/local/include
                $ENV{HOME}/mlx/include
                $ENV{MLX_ROOT}/include
            NO_DEFAULT_PATH
        )

        find_library(MLX_LIBRARY
            NAMES mlx libmlx
            PATHS
                ${CMAKE_SOURCE_DIR}/.venv/lib/python3.12/site-packages/mlx/lib
                ${CMAKE_SOURCE_DIR}/.venv/lib/python3.11/site-packages/mlx/lib
                /opt/homebrew/lib
                /usr/local/lib
                $ENV{HOME}/mlx/lib
                $ENV{MLX_ROOT}/lib
            NO_DEFAULT_PATH
        )

        if(MLX_INCLUDE_DIR AND MLX_LIBRARY)
            set(MLX_FOUND TRUE)
            message(STATUS "Found MLX: ${MLX_LIBRARY}")
            message(STATUS "MLX include: ${MLX_INCLUDE_DIR}")
        else()
            message(STATUS "MLX not found. To install: pip install mlx")
            message(STATUS "Or build from source: https://github.com/ml-explore/mlx")
            message(WARNING "Disabling MLX backend.")
            set(WITH_MLX OFF PARENT_SCOPE)
            return()
        endif()
    endif()

    # Standard C++ source files
    set(MLX_BACKEND_SOURCES
        mlx_backend.cpp
        fhe.cpp
        twiddle_cache.cpp
    )

    # Objective-C++ source files (native Metal API)
    set(MLX_OBJCPP_SOURCES
        metal_ntt_wrapper.mm
    )

    # Create the MLX backend library
    add_library(FHEmlx OBJECT ${MLX_BACKEND_SOURCES})

    # Create native Metal library (Objective-C++)
    add_library(FHEmetal OBJECT ${MLX_OBJCPP_SOURCES})
    set_source_files_properties(${MLX_OBJCPP_SOURCES} PROPERTIES
        COMPILE_FLAGS "-x objective-c++ -fobjc-arc"
    )
    target_include_directories(FHEmetal PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${MLX_INCLUDE_DIR}
    )
    target_link_libraries(FHEmetal PUBLIC
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
    )
    target_compile_definitions(FHEmetal PUBLIC WITH_MLX WITH_NATIVE_METAL)
    target_compile_features(FHEmetal PUBLIC cxx_std_17)
    
    target_include_directories(FHEmlx PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${MLX_INCLUDE_DIR}
    )
    
    target_link_libraries(FHEmlx PUBLIC
        ${MLX_LIBRARY}
    )
    
    # Add compile definitions
    target_compile_definitions(FHEmlx PUBLIC WITH_MLX)
    
    # C++17 required for MLX
    target_compile_features(FHEmlx PUBLIC cxx_std_17)
    
    # Export the targets and libraries
    # Export targets as separate variables for proper CMake handling
    set(MLX_BACKEND_TARGET FHEmlx PARENT_SCOPE)
    set(MLX_METAL_TARGET FHEmetal PARENT_SCOPE)
    set(MLX_LIBS "${MLX_LIBRARY};${METAL_FRAMEWORK};${FOUNDATION_FRAMEWORK}" PARENT_SCOPE)
    set(MLX_INCLUDE_DIRS ${MLX_INCLUDE_DIR} PARENT_SCOPE)

    # NTT Test executable (MLX)
    add_executable(ntt_test ntt_test.cpp)
    target_include_directories(ntt_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${MLX_INCLUDE_DIR}
    )
    target_link_libraries(ntt_test PRIVATE ${MLX_LIBRARY})
    target_compile_definitions(ntt_test PRIVATE WITH_MLX)
    target_compile_features(ntt_test PRIVATE cxx_std_17)

    # Native Metal NTT Benchmark (Objective-C++)
    add_executable(metal_ntt_bench metal_bench.mm)
    set_source_files_properties(metal_bench.mm PROPERTIES
        COMPILE_FLAGS "-x objective-c++ -fobjc-arc"
    )
    target_include_directories(metal_ntt_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include
    )
    target_link_libraries(metal_ntt_bench PRIVATE
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
    )
    target_compile_definitions(metal_ntt_bench PRIVATE WITH_NATIVE_METAL)
    target_compile_features(metal_ntt_bench PRIVATE cxx_std_17)

    # euint256 Test executable
    add_executable(euint256_test euint256_test.cpp)
    target_include_directories(euint256_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include
        ${MLX_INCLUDE_DIR}
    )
    target_link_libraries(euint256_test PRIVATE ${MLX_LIBRARY})
    target_compile_definitions(euint256_test PRIVATE WITH_MLX)
    target_compile_features(euint256_test PRIVATE cxx_std_17)

    # Add to test suite
    enable_testing()
    add_test(NAME NTTTest COMMAND ntt_test)
    add_test(NAME euint256Test COMMAND euint256_test)
    add_test(NAME MetalNTTBench COMMAND metal_ntt_bench)

    message(STATUS "MLX GPU Backend: Configured successfully")
    message(STATUS "Native Metal NTT: Enabled (10x+ speedup for N<=4096)")
else()
    message(STATUS "MLX GPU Backend: Disabled (use -DWITH_MLX=ON to enable)")
endif()
