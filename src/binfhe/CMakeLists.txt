#
# CMakeLists.txt for BINFHE library
#

# all files named *.cpp are compiled to form the library
file(GLOB BINFHE_SRC_FILES CONFIGURE_DEPENDS lib/*.cpp)

# Lux FHE extensions: batch APIs, radix integers, backend abstraction, fhEVM, security
option(WITH_LUX_EXTENSIONS "Build Lux FHE extensions (batch, radix, backend, fhevm, security)" ON)
if(WITH_LUX_EXTENSIONS)
    message(STATUS "Lux FHE Extensions: Enabled")
    file(GLOB BINFHE_BATCH_SRC_FILES CONFIGURE_DEPENDS lib/batch/*.cpp)
    file(GLOB BINFHE_RADIX_SRC_FILES CONFIGURE_DEPENDS lib/radix/*.cpp)
    file(GLOB BINFHE_BACKEND_SRC_FILES CONFIGURE_DEPENDS lib/backend/*.cpp)
    file(GLOB BINFHE_FHEVM_SRC_FILES CONFIGURE_DEPENDS lib/fhevm/*.cpp)
    file(GLOB BINFHE_SECURITY_SRC_FILES CONFIGURE_DEPENDS lib/security/*.cpp)
    file(GLOB BINFHE_THRESHOLD_SRC_FILES CONFIGURE_DEPENDS lib/threshold/*.cpp)
    list(APPEND BINFHE_SRC_FILES ${BINFHE_BATCH_SRC_FILES})
    list(APPEND BINFHE_SRC_FILES ${BINFHE_RADIX_SRC_FILES})
    list(APPEND BINFHE_SRC_FILES ${BINFHE_BACKEND_SRC_FILES})
    list(APPEND BINFHE_SRC_FILES ${BINFHE_FHEVM_SRC_FILES})
    list(APPEND BINFHE_SRC_FILES ${BINFHE_SECURITY_SRC_FILES})
    list(APPEND BINFHE_SRC_FILES ${BINFHE_THRESHOLD_SRC_FILES})
else()
    message(STATUS "Lux FHE Extensions: Disabled (use -DWITH_LUX_EXTENSIONS=ON to enable)")
endif()

include_directories(${CORE_INCLUDE_DIRS})
list(APPEND BINFHE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
list(APPEND BINFHE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/lib")
if(WITH_LUX_EXTENSIONS)
    list(APPEND BINFHE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include/batch")
    list(APPEND BINFHE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include/radix")
    list(APPEND BINFHE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include/backend")
    list(APPEND BINFHE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include/fhevm")
    list(APPEND BINFHE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include/security")
    list(APPEND BINFHE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include/threshold")
endif()
include_directories(${BINFHE_INCLUDE_DIRS})

set(BINFHE_VERSION_MAJOR ${OPENFHE_VERSION_MAJOR})
set(BINFHE_VERSION_MINOR ${OPENFHE_VERSION_MINOR})
set(BINFHE_VERSION_PATCH ${OPENFHE_VERSION_PATCH})
set(BINFHE_VERSION ${BINFHE_VERSION_MAJOR}.${BINFHE_VERSION_MINOR}.${BINFHE_VERSION_PATCH})

# =============================================================================
# Metal TFHE Shader Compilation (macOS)
# =============================================================================
set(BINFHE_METAL_RESOURCES)
if(APPLE AND WITH_GPU)
    find_program(XCRUN xcrun)
    if(XCRUN)
        # TFHE Metal shaders for GPU-accelerated bootstrapping
        set(TFHE_SHADERS
            tfhe_bootstrap
            tfhe_external_product
            tfhe_keyswitch
            tfhe_sample_extract
            blind_rotate
        )
        set(TFHE_METALLIB "${CMAKE_CURRENT_BINARY_DIR}/lux_tfhe.metallib")
        set(TFHE_AIR_FILES)

        # Compile each .metal to .air
        foreach(SHADER ${TFHE_SHADERS})
            set(SHADER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/lib/backend/metal/${SHADER}.metal")
            set(SHADER_AIR "${CMAKE_CURRENT_BINARY_DIR}/${SHADER}.air")
            if(EXISTS "${SHADER_SRC}")
                add_custom_command(
                    OUTPUT ${SHADER_AIR}
                    COMMAND ${XCRUN} -sdk macosx metal -c ${SHADER_SRC} -o ${SHADER_AIR}
                    DEPENDS ${SHADER_SRC}
                    COMMENT "Compiling TFHE Metal shader: ${SHADER}.metal"
                )
                list(APPEND TFHE_AIR_FILES ${SHADER_AIR})
                message(STATUS "TFHE Metal shader: ${SHADER}.metal")
            endif()
        endforeach()

        # Link all .air files into single .metallib
        if(TFHE_AIR_FILES)
            add_custom_command(
                OUTPUT ${TFHE_METALLIB}
                COMMAND ${XCRUN} -sdk macosx metallib ${TFHE_AIR_FILES} -o ${TFHE_METALLIB}
                DEPENDS ${TFHE_AIR_FILES}
                COMMENT "Linking TFHE Metal library: lux_tfhe.metallib"
            )
            add_custom_target(metal_tfhe_shaders DEPENDS ${TFHE_METALLIB})
            set(BINFHE_METAL_RESOURCES ${TFHE_METALLIB})
        endif()
    endif()
endif()

add_library(binfheobj OBJECT ${BINFHE_SRC_FILES})
set_property(TARGET binfheobj PROPERTY POSITION_INDEPENDENT_CODE 1)

if(BUILD_SHARED)
    add_dependencies(binfheobj FHEcore)
    add_library(FHEbin SHARED $<TARGET_OBJECTS:binfheobj>)
    set_property(TARGET FHEbin PROPERTY PREFIX "")
    set_property(TARGET FHEbin PROPERTY VERSION ${BINFHE_VERSION})
    set_property(TARGET FHEbin PROPERTY SOVERSION ${BINFHE_VERSION_MAJOR})
    set_property(TARGET FHEbin PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    install(TARGETS FHEbin EXPORT OpenFHETargets DESTINATION lib)
    # Add Metal shader dependency
    if(TARGET metal_tfhe_shaders)
        add_dependencies(FHEbin metal_tfhe_shaders)
    endif()
endif()

if(BUILD_STATIC)
add_dependencies(binfheobj FHEcore_static)
    add_library(FHEbin_static STATIC $<TARGET_OBJECTS:binfheobj>)
    set_property(TARGET FHEbin_static PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    install(TARGETS FHEbin_static EXPORT OpenFHETargets DESTINATION lib)
endif()

install(DIRECTORY include/ DESTINATION include/openfhe/binfhe)

add_custom_target(allbinfhe)

if(BUILD_SHARED)
set(BINFHELIBS PUBLIC FHEbin PUBLIC FHEcore ${THIRDPARTYLIBS} ${OpenMP_CXX_FLAGS})
    target_link_libraries(FHEbin PUBLIC FHEcore ${THIRDPARTYLIBS} ${OpenMP_CXX_FLAGS} ${ADDITIONAL_LIBS})
    add_dependencies(allbinfhe FHEbin)
endif()

if(BUILD_STATIC)
set(BINFHELIBS ${BINFHELIBS} PUBLIC FHEbin_static PUBLIC FHEcore_static ${THIRDPARTYSTATICLIBS} ${OpenMP_CXX_FLAGS})
    target_link_libraries(FHEbin_static PUBLIC FHEcore_static ${THIRDPARTYSTATICLIBS} ${OpenMP_CXX_FLAGS} ${ADDITIONAL_LIBS})
    add_dependencies(allbinfhe FHEbin_static)
endif()

if(BUILD_UNITTESTS)
    file(GLOB BINFHE_TEST_SRC_FILES CONFIGURE_DEPENDS unittest/*.cpp)
    add_executable(binfhe_tests ${BINFHE_TEST_SRC_FILES} ${UNITTESTMAIN})
    set_property(TARGET binfhe_tests PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unittest)
    target_link_libraries(binfhe_tests PRIVATE gtest gtest_main)
    target_link_libraries(binfhe_tests ${BINFHELIBS} ${ADDITIONAL_LIBS})
    if(NOT ${WITH_OPENMP} AND NOT EMSCRIPTEN)
        target_link_libraries(binfhe_tests PRIVATE Threads::Threads)
    endif()

    add_dependencies(allbinfhe binfhe_tests)

    add_custom_command(OUTPUT runbinfhetests WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND ${CMAKE_BINARY_DIR}/unittest/binfhe_tests)
    add_custom_target(testbinfhe DEPENDS binfhe_tests runbinfhetests)
endif()

set(BINFHEAPPS "")
if(BUILD_EXAMPLES)
    file(GLOB BINFHE_EXAMPLES_SRC_FILES CONFIGURE_DEPENDS examples/*.cpp)
    foreach(app ${BINFHE_EXAMPLES_SRC_FILES})
        get_filename_component(exe ${app} NAME_WE)
        add_executable(${exe} ${app})
        set_property(TARGET ${exe} PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/examples/binfhe)
        set(BINFHEAPPS ${BINFHEAPPS} ${exe})
        target_link_libraries(${exe} ${BINFHELIBS} ${ADDITIONAL_LIBS})
    endforeach()

    file(GLOB BINFHE_EXAMPLES_SRC_FILES CONFIGURE_DEPENDS examples/pke/*.cpp)
    foreach(app ${BINFHE_EXAMPLES_SRC_FILES})
        get_filename_component(exe ${app} NAME_WE)
        add_executable(${exe} ${app})
        set_property(TARGET ${exe} PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/examples/binfhe/pke)
        set(BINFHEAPPS ${BINFHEAPPS} ${exe})
        target_link_libraries(${exe} ${BINFHELIBS} ${ADDITIONAL_LIBS})
    endforeach()

    add_custom_target(allbinfheexamples)
    add_dependencies(allbinfheexamples ${BINFHEAPPS})
    add_dependencies(allbinfhe allbinfheexamples)
endif()

add_custom_command(OUTPUT binfheinfocmd COMMAND echo Builds FHEbin and these apps: ${BINFHEAPPS})
add_custom_target(binfheinfo DEPENDS binfheinfocmd)

# Collect compile definitions and pass them upward
if(BUILD_SHARED)
    get_target_property(_compile_defs FHEbin COMPILE_DEFINITIONS)
    set(_pal_binfhe_compile_defs ${_compile_defs} PARENT_SCOPE)
endif()

if(BUILD_STATIC)
    get_target_property(_compile_defs_static FHEbin_static COMPILE_DEFINITIONS)
    set(_pal_binfhe_compile_defs_static ${_compile_defs_static} PARENT_SCOPE)
endif()
