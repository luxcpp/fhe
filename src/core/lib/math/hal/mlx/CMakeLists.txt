# FHE GPU Backend CMakeLists.txt
#
# GPU acceleration via lux::gpu (Metal/CUDA/CPU)
# Enable with: cmake -DWITH_GPU=ON ..
#
# All GPU operations go through lux::gpu which provides:
# - Metal (Apple Silicon)
# - CUDA (NVIDIA)
# - CPU fallback (all platforms)

# Support both WITH_GPU and WITH_MLX for backwards compatibility
if(WITH_MLX)
    set(WITH_GPU ON)
endif()

if(WITH_GPU)
    message(STATUS "FHE GPU Backend: Enabled")

    # Platform-specific framework requirements
    if(APPLE)
        find_library(METAL_FRAMEWORK Metal REQUIRED)
        find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
        message(STATUS "Found Metal framework: ${METAL_FRAMEWORK}")
    endif()

    # Find lux-gpu (provides GPU acceleration: Metal/CUDA/CPU)
    # Priority: CMAKE_INSTALL_PREFIX > luxcpp/install > system
    # Note: find_package should be called at parent scope for target visibility
    if(TARGET lux::gpu)
        set(GPU_FOUND TRUE)
        set(GPU_LIBRARY lux::gpu)
        set(GPU_USE_TARGET TRUE)
        message(STATUS "Using lux::gpu target from parent scope")
    elseif(lux-gpu_FOUND)
        set(GPU_FOUND TRUE)
        set(GPU_LIBRARY lux::gpu)
        set(GPU_USE_TARGET TRUE)
        message(STATUS "Found lux::gpu via CMake config")
    else()
        # Fallback: manual search for libluxgpu
        find_path(GPU_INCLUDE_DIR
            NAMES mlx/mlx.h lux/gpu/gpu.h
            PATHS
                ${CMAKE_INSTALL_PREFIX}/include
                ${CMAKE_SOURCE_DIR}/../install/include
                /opt/homebrew/include
                /usr/local/include
                $ENV{LUX_GPU_ROOT}/include
            NO_DEFAULT_PATH
        )

        find_library(GPU_LIBRARY
            NAMES luxgpu lux_gpu mlx
            PATHS
                ${CMAKE_INSTALL_PREFIX}/lib
                ${CMAKE_SOURCE_DIR}/../install/lib
                /opt/homebrew/lib
                /usr/local/lib
                $ENV{LUX_GPU_ROOT}/lib
            NO_DEFAULT_PATH
        )

        if(GPU_INCLUDE_DIR AND GPU_LIBRARY)
            set(GPU_FOUND TRUE)
            message(STATUS "Found lux-gpu: ${GPU_LIBRARY}")
            message(STATUS "GPU include: ${GPU_INCLUDE_DIR}")
        else()
            message(STATUS "lux-gpu not found. Build luxcpp/gpu first:")
            message(STATUS "  cd ../gpu && cmake -B build -DCMAKE_INSTALL_PREFIX=../install && cmake --build build && cmake --install build")
            message(WARNING "Disabling GPU backend.")
            set(WITH_GPU OFF PARENT_SCOPE)
            return()
        endif()
    endif()

    # C++ source files
    set(GPU_CPP_SOURCES
        mlx_backend.cpp
        fhe.cpp
        twiddle_cache.cpp
    )

    # Platform-specific sources
    if(APPLE)
        # Objective-C++ source files (native Metal dispatch)
        set(GPU_OBJCPP_SOURCES
            metal_ntt_wrapper.mm
        )
    endif()

    # Create unified FHEgpu library (combines C++ and Obj-C++ sources)
    if(APPLE)
        add_library(FHEgpu OBJECT ${GPU_CPP_SOURCES} ${GPU_OBJCPP_SOURCES})
        # Set Obj-C++ flags for .mm files
        set_source_files_properties(${GPU_OBJCPP_SOURCES} PROPERTIES
            COMPILE_FLAGS "-x objective-c++ -fobjc-arc"
        )
    else()
        add_library(FHEgpu OBJECT ${GPU_CPP_SOURCES})
    endif()

    target_include_directories(FHEgpu PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # Link lux::gpu and platform frameworks
    if(GPU_USE_TARGET)
        target_link_libraries(FHEgpu PUBLIC ${GPU_LIBRARY})
    else()
        target_include_directories(FHEgpu PUBLIC ${GPU_INCLUDE_DIR})
        target_link_libraries(FHEgpu PUBLIC ${GPU_LIBRARY})
    endif()

    if(APPLE)
        target_link_libraries(FHEgpu PUBLIC ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
    endif()

    # Add compile definitions
    # WITH_MLX retained for backwards compatibility with source files that check #ifdef WITH_MLX
    target_compile_definitions(FHEgpu PUBLIC WITH_GPU WITH_MLX)
    if(APPLE)
        target_compile_definitions(FHEgpu PUBLIC WITH_NATIVE_METAL)
    endif()

    # C++17 required for GPU backend
    target_compile_features(FHEgpu PUBLIC cxx_std_17)

    # Export the GPU target
    set(GPU_BACKEND_TARGET FHEgpu PARENT_SCOPE)
    if(APPLE)
        set(GPU_LIBS "${GPU_LIBRARY};${METAL_FRAMEWORK};${FOUNDATION_FRAMEWORK}" PARENT_SCOPE)
    else()
        set(GPU_LIBS "${GPU_LIBRARY}" PARENT_SCOPE)
    endif()
    set(GPU_INCLUDE_DIRS ${GPU_INCLUDE_DIR} PARENT_SCOPE)

    # NTT Test executable (lux::gpu)
    add_executable(ntt_test ntt_test.cpp)
    target_include_directories(ntt_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    if(GPU_USE_TARGET)
        target_link_libraries(ntt_test PRIVATE ${GPU_LIBRARY})
    else()
        target_include_directories(ntt_test PRIVATE ${GPU_INCLUDE_DIR})
        target_link_libraries(ntt_test PRIVATE ${GPU_LIBRARY})
    endif()
    target_compile_definitions(ntt_test PRIVATE WITH_GPU)
    target_compile_features(ntt_test PRIVATE cxx_std_17)

    # euint256 Test executable
    add_executable(euint256_test euint256_test.cpp)
    target_include_directories(euint256_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include
    )
    if(GPU_USE_TARGET)
        target_link_libraries(euint256_test PRIVATE ${GPU_LIBRARY})
    else()
        target_include_directories(euint256_test PRIVATE ${GPU_INCLUDE_DIR})
        target_link_libraries(euint256_test PRIVATE ${GPU_LIBRARY})
    endif()
    target_compile_definitions(euint256_test PRIVATE WITH_GPU)
    target_compile_features(euint256_test PRIVATE cxx_std_17)

    # Platform-specific tests
    if(APPLE)
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
    endif()

    # Add to test suite
    enable_testing()
    add_test(NAME NTTTest COMMAND ntt_test)
    add_test(NAME euint256Test COMMAND euint256_test)
    if(APPLE)
        add_test(NAME MetalNTTBench COMMAND metal_ntt_bench)
    endif()

    message(STATUS "GPU Backend: Configured successfully (lux::gpu)")
    if(APPLE)
        message(STATUS "Native Metal NTT: Enabled (10x+ speedup for N<=4096)")
    endif()
else()
    message(STATUS "GPU Backend: Disabled (use -DWITH_GPU=ON to enable)")
endif()
