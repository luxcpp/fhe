# MLX GPU Backend CMakeLists.txt
#
# Optional GPU acceleration using Apple's MLX framework
# Enable with: cmake -DWITH_MLX=ON ..

if(WITH_MLX)
    message(STATUS "MLX GPU Backend: Enabled")
    
    # Check for Apple Silicon
    if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        message(WARNING "MLX is only supported on Apple Silicon (arm64). Disabling MLX backend.")
        set(WITH_MLX OFF PARENT_SCOPE)
        return()
    endif()
    
    # Find MLX
    find_package(MLX CONFIG QUIET)
    
    if(NOT MLX_FOUND)
        # Try to find MLX in common locations
        # First check venv in project directory
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
    
    # Add source files
    set(MLX_BACKEND_SOURCES
        mlx_backend.cpp
        gpu_tfhe.cpp
    )
    
    # Metal shader sources (compiled separately)
    set(MLX_METAL_SOURCES
        tfhe_kernels.metal
    )
    
    # Create the MLX backend library
    add_library(OPENFHEmlx OBJECT ${MLX_BACKEND_SOURCES})
    
    target_include_directories(OPENFHEmlx PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include
        ${MLX_INCLUDE_DIR}
    )
    
    target_link_libraries(OPENFHEmlx PUBLIC
        ${MLX_LIBRARY}
    )
    
    # Add compile definitions
    target_compile_definitions(OPENFHEmlx PUBLIC WITH_MLX)
    
    # C++17 required for MLX
    target_compile_features(OPENFHEmlx PUBLIC cxx_std_17)
    
    # Export the target and libraries
    set(MLX_BACKEND_TARGET OPENFHEmlx PARENT_SCOPE)
    set(MLX_LIBS ${MLX_LIBRARY} PARENT_SCOPE)
    set(MLX_INCLUDE_DIRS ${MLX_INCLUDE_DIR} PARENT_SCOPE)
    
    message(STATUS "MLX GPU Backend: Configured successfully")
else()
    message(STATUS "MLX GPU Backend: Disabled (use -DWITH_MLX=ON to enable)")
endif()
