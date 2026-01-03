#
# CMakeLists.txt for CORE library
#

# all files named *.c or */cpp are compiled to form the library
# Exclude MLX GPU files - they are conditionally added below via WITH_MLX
file(GLOB_RECURSE CORE_SRC_FILES CONFIGURE_DEPENDS lib/*.c lib/*.cpp lib/utils/*.cpp)
list(FILTER CORE_SRC_FILES EXCLUDE REGEX ".*/mlx/.*")

# GPU Backend (optional) - uses lux::gpu (MLX/Metal/CUDA/CPU)
if(WITH_GPU OR WITH_MLX)
    set(WITH_GPU ON)  # Normalize to WITH_GPU

    # Find lux-gpu at parent scope so imported target is visible
    find_package(lux-gpu CONFIG QUIET
        HINTS
            ${CMAKE_INSTALL_PREFIX}/share/cmake/lux-gpu
            ${CMAKE_SOURCE_DIR}/../install/share/cmake/lux-gpu
    )
    if(lux-gpu_FOUND)
        message(STATUS "Found lux::gpu via CMake config")
    endif()

    add_subdirectory(lib/math/hal/mlx)
    if(GPU_BACKEND_TARGET)
        # Note: GPU objects are added directly to FHEcore/FHEcore_static below
        if(GPU_INCLUDE_DIRS)
            list(APPEND CORE_INCLUDE_DIRS ${GPU_INCLUDE_DIRS})
        endif()
        add_compile_definitions(WITH_GPU)
        message(STATUS "GPU Backend included in core library")
    endif()
endif()

list(APPEND CORE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
list(APPEND CORE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/lib")
include_directories(${CORE_INCLUDE_DIRS})
set(CORE_INCLUDE_DIRS "${CORE_INCLUDE_DIRS}" CACHE INTERNAL "")

set(CORE_VERSION_MAJOR ${OPENFHE_VERSION_MAJOR})
set(CORE_VERSION_MINOR ${OPENFHE_VERSION_MINOR})
set(CORE_VERSION_PATCH ${OPENFHE_VERSION_PATCH})
set(CORE_VERSION ${CORE_VERSION_MAJOR}.${CORE_VERSION_MINOR}.${CORE_VERSION_PATCH})

add_library(coreobj OBJECT ${CORE_SRC_FILES})
add_dependencies(coreobj third-party)

set_property(TARGET coreobj PROPERTY POSITION_INDEPENDENT_CODE 1)

if(BUILD_SHARED)
    # Build shared library with core objects
    set(CORE_OBJECT_LIBS $<TARGET_OBJECTS:coreobj>)
    # Add GPU objects if enabled (unified FHEgpu target)
    if(WITH_GPU AND GPU_BACKEND_TARGET)
        list(APPEND CORE_OBJECT_LIBS $<TARGET_OBJECTS:${GPU_BACKEND_TARGET}>)
    endif()
    add_library(FHEcore SHARED ${CORE_OBJECT_LIBS})
    set_property(TARGET FHEcore PROPERTY VERSION ${CORE_VERSION})
    set_property(TARGET FHEcore PROPERTY SOVERSION ${CORE_VERSION_MAJOR})
    set_property(TARGET FHEcore PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    install(TARGETS FHEcore EXPORT OpenFHETargets DESTINATION lib)
endif()

if(BUILD_STATIC)
    # Build static library with core objects
    set(CORE_STATIC_OBJECT_LIBS $<TARGET_OBJECTS:coreobj>)
    # Add GPU objects if enabled (unified FHEgpu target)
    if(WITH_GPU AND GPU_BACKEND_TARGET)
        list(APPEND CORE_STATIC_OBJECT_LIBS $<TARGET_OBJECTS:${GPU_BACKEND_TARGET}>)
    endif()
    add_library(FHEcore_static STATIC ${CORE_STATIC_OBJECT_LIBS})
    set_property(TARGET FHEcore_static PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    install(TARGETS FHEcore_static EXPORT OpenFHETargets DESTINATION lib)
endif()

install(DIRECTORY include/ DESTINATION include/openfhe/core)

add_custom_target(allcore)

if(BUILD_SHARED)
    set(CORELIBS PUBLIC FHEcore ${THIRDPARTYLIBS} ${OpenMP_CXX_FLAGS})
    target_link_libraries(FHEcore ${THIRDPARTYLIBS} ${OpenMP_CXX_FLAGS} ${ADDITIONAL_LIBS})
    # Link GPU library if enabled
    if(WITH_GPU AND TARGET lux::gpu)
        target_link_libraries(FHEcore lux::gpu)
        if(APPLE)
            find_library(METAL_FRAMEWORK Metal)
            find_library(FOUNDATION_FRAMEWORK Foundation)
            if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
                target_link_libraries(FHEcore ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
            endif()
        endif()
    elseif(WITH_GPU AND GPU_LIBS)
        # Fallback to library path
        target_link_libraries(FHEcore ${GPU_LIBS})
    endif()
    add_dependencies(allcore FHEcore)
endif()

if(BUILD_STATIC)
    set(CORELIBS ${CORELIBS} PUBLIC FHEcore_static ${THIRDPARTYSTATICLIBS} ${OpenMP_CXX_FLAGS})
    target_link_libraries(FHEcore_static ${THIRDPARTYSTATICLIBS} ${OpenMP_CXX_FLAGS} ${ADDITIONAL_LIBS})
    add_dependencies(allcore FHEcore_static)
endif()

if(BUILD_UNITTESTS)
    if("${NATIVE_SIZE}" EQUAL 32)
        message("**** core_tests are not linked for NATIVE_SIZE=32")
    else()
        file(GLOB CORE_TEST_SRC_FILES CONFIGURE_DEPENDS unittest/*.cpp)
        # Exclude MLX-specific tests when MLX is disabled
        if(NOT WITH_MLX)
            list(FILTER CORE_TEST_SRC_FILES EXCLUDE REGEX ".*MLX.*|.*GPUTFHE.*")
        endif()
    endif()
    set(CORE_TEST_SRC_FILES ${CORE_TEST_SRC_FILES})
    add_executable(core_tests ${CORE_TEST_SRC_FILES} ${UNITTESTMAIN})
    set_property(TARGET core_tests PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unittest)
    target_link_libraries(core_tests PRIVATE gtest gtest_main)
    target_link_libraries(core_tests ${CORELIBS} ${ADDITIONAL_LIBS})
    if(NOT ${WITH_OPENMP} AND NOT EMSCRIPTEN)
        target_link_libraries(core_tests PRIVATE Threads::Threads)
    endif()
    add_dependencies(allcore core_tests)
    add_custom_command(OUTPUT runcoretests WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND ${CMAKE_BINARY_DIR}/unittest/core_tests)
    add_custom_target(testcore DEPENDS core_tests runcoretests)
endif()

set(COREAPPS "")
if(BUILD_EXAMPLES)
    file(GLOB CORE_EXAMPLES_SRC_FILES CONFIGURE_DEPENDS examples/*.cpp)
    foreach(app ${CORE_EXAMPLES_SRC_FILES})
        get_filename_component(exe ${app} NAME_WE)
        if(${exe} STREQUAL "parallel" AND NOT ${WITH_OPENMP})
            message("Skipping ${exe} because WITH_OPENMP=OFF")
            continue()
        endif()
        add_executable(${exe} ${app})
        set_property(TARGET ${exe} PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/examples/core)
        set(COREAPPS ${COREAPPS} ${exe})
        target_link_libraries(${exe} ${CORELIBS} ${ADDITIONAL_LIBS})
    endforeach()
    add_custom_target(allcoreexamples)
    add_dependencies(allcoreexamples ${COREAPPS})
    add_dependencies(allcore allcoreexamples)
endif()

set(COREEXTRAS "")
if(BUILD_EXTRAS)
    file(GLOB CORE_EXTRAS_SRC_FILES CONFIGURE_DEPENDS extras/*.cpp)
    foreach(app ${CORE_EXTRAS_SRC_FILES})
        get_filename_component(exe ${app} NAME_WE)
        add_executable(${exe} ${app})
        set_property(TARGET ${exe} PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/extras/core)
        set(COREEXTRAS ${COREEXTRAS} ${exe})
        target_link_libraries(${exe} ${CORELIBS} ${ADDITIONAL_LIBS})
    endforeach()
    add_custom_target(allcoreextras)
    add_dependencies(allcoreextras ${COREEXTRAS})
    add_dependencies(allcore allcoreextras)
endif()

add_custom_command(OUTPUT coreinfocmd COMMAND echo Builds FHEcore and these apps: ${COREAPPS})
add_custom_target(coreinfo DEPENDS coreinfocmd)

# Collect compile definitions and pass them upward
if(BUILD_SHARED)
    get_target_property(_compile_defs FHEcore COMPILE_DEFINITIONS)
    set(_pal_core_compile_defs ${_compile_defs} PARENT_SCOPE)
endif()

if(BUILD_STATIC)
    get_target_property(_compile_defs_static FHEcore_static COMPILE_DEFINITIONS)
    set(_pal_core_compile_defs_static ${_compile_defs_static} PARENT_SCOPE)
endif()
