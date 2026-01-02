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

    # Find lux-gpu (provides MLX/Metal GPU acceleration)
    # Priority: CMAKE_INSTALL_PREFIX > luxcpp/install > system
    find_package(lux-gpu CONFIG QUIET
        HINTS
            ${CMAKE_INSTALL_PREFIX}
            ${CMAKE_SOURCE_DIR}/../install
    )

    if(lux-gpu_FOUND)
        set(MLX_FOUND TRUE)
        set(MLX_LIBRARY lux::gpu)
        set(MLX_USE_TARGET TRUE)  # Use target for includes instead of MLX_INCLUDE_DIR
        message(STATUS "Found lux-gpu via CMake config")
    else()
        # Fallback: manual search for liblux_gpu or libmlx
        find_path(MLX_INCLUDE_DIR
            NAMES mlx/mlx.h lux/gpu/mlx.h
            PATHS
                ${CMAKE_INSTALL_PREFIX}/include
                ${CMAKE_SOURCE_DIR}/../install/include/lux/gpu
                ${CMAKE_SOURCE_DIR}/../gpu/mlx
                /opt/homebrew/include
                /usr/local/include
                $ENV{MLX_ROOT}/include
            NO_DEFAULT_PATH
        )

        find_library(MLX_LIBRARY
            NAMES lux_gpu mlx
            PATHS
                ${CMAKE_INSTALL_PREFIX}/lib
                ${CMAKE_SOURCE_DIR}/../install/lib
                /opt/homebrew/lib
                /usr/local/lib
                $ENV{MLX_ROOT}/lib
            NO_DEFAULT_PATH
        )

        if(MLX_INCLUDE_DIR AND MLX_LIBRARY)
            set(MLX_FOUND TRUE)
            message(STATUS "Found MLX: ${MLX_LIBRARY}")
            message(STATUS "MLX include: ${MLX_INCLUDE_DIR}")
        else()
            message(STATUS "lux-gpu not found. Build luxcpp/gpu first:")
            message(STATUS "  cd ../gpu && cmake -B build -DCMAKE_INSTALL_PREFIX=../install && cmake --build build && cmake --install build")
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
    )
    if(MLX_USE_TARGET)
        target_link_libraries(FHEmetal PUBLIC lux::gpu ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
    else()
        target_include_directories(FHEmetal PUBLIC ${MLX_INCLUDE_DIR})
        target_link_libraries(FHEmetal PUBLIC ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
    endif()
    target_compile_definitions(FHEmetal PUBLIC WITH_MLX WITH_NATIVE_METAL)
    target_compile_features(FHEmetal PUBLIC cxx_std_17)
    
    target_include_directories(FHEmlx PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    if(MLX_USE_TARGET)
        target_link_libraries(FHEmlx PUBLIC lux::gpu)
    else()
        target_include_directories(FHEmlx PUBLIC ${MLX_INCLUDE_DIR})
        target_link_libraries(FHEmlx PUBLIC ${MLX_LIBRARY})
    endif()
    
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
