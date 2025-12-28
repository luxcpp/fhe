# - Config file for the OpenFHE package
# It defines the following variables
#  OpenFHE_INCLUDE_DIRS - include directories for OpenFHE
#  OpenFHE_LIBRARIES    - libraries to link against
get_filename_component(OpenFHE_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

# Our library dependencies (contains definitions for IMPORTED targets)
if(NOT OpenFHE_BINARY_DIR)
    include("${OpenFHE_CMAKE_DIR}/OpenFHETargets.cmake")
endif()

# These are IMPORTED targets created by OpenFHETargets.cmake
# set(OpenFHE_INCLUDE "${OpenFHE_CMAKE_DIR}/../../include/openfhe")
# set(OpenFHE_LIBDIR "${OpenFHE_CMAKE_DIR}/../../lib")
set(OpenFHE_INCLUDE "/Users/z/work/lux/fhe/install/include/openfhe")
set(OpenFHE_LIBDIR "/Users/z/work/lux/fhe/install/lib")
set(OpenFHE_LIBRARIES OPENFHEcore;OPENFHEpke;OPENFHEbinfhe  -Xpreprocessor -fopenmp -lomp -Wno-unused-command-line-argument)
set(OpenFHE_STATIC_LIBRARIES   -Xpreprocessor -fopenmp -lomp -Wno-unused-command-line-argument)
set(OpenFHE_SHARED_LIBRARIES OPENFHEcore;OPENFHEpke;OPENFHEbinfhe  -Xpreprocessor -fopenmp -lomp -Wno-unused-command-line-argument)
set(BASE_OPENFHE_VERSION 1.4.2)

set(OPENMP_INCLUDES "/opt/homebrew/opt/libomp/include")
set(OPENMP_LIBRARIES "/opt/homebrew/opt/libomp/lib")

set(OpenFHE_CXX_FLAGS " -Wall -Werror -DOPENFHE_VERSION=1.4.2 -O3 -DMATHBACKEND=4 -Xpreprocessor -fopenmp -lomp -Wno-unused-command-line-argument -Xpreprocessor -fopenmp -lomp -Wno-unused-command-line-argument")
set(OpenFHE_C_FLAGS " -Wall -Werror -DOPENFHE_VERSION=1.4.2 -O3 -DMATHBACKEND=4 -Xpreprocessor -fopenmp -lomp -Wno-unused-command-line-argument -Xpreprocessor -fopenmp -lomp -Wno-unused-command-line-argument")

set(OpenFHE_EXE_LINKER_FLAGS " ")

# CXX info
set(OpenFHE_CXX_STANDARD "17")
set(OpenFHE_CXX_COMPILER_ID "AppleClang")
set(OpenFHE_CXX_COMPILER_VERSION "17.0.0.17000319")

# Build Options
set(OpenFHE_STATIC "OFF")
set(OpenFHE_SHARED "ON")
set(OpenFHE_TCM "OFF")
set(OpenFHE_NTL "OFF")
set(OpenFHE_OPENMP "ON")
set(OpenFHE_NATIVE_SIZE "64")
set(OpenFHE_CKKS_M_FACTOR "1")
set(OpenFHE_NATIVEOPT "OFF")
set(OpenFHE_NOISEDEBUG "OFF")
set(OpenFHE_REDUCEDNOISE "OFF")

# Math Backend
set(OpenFHE_BACKEND "4")

# Build Details
set(OpenFHE_EMSCRIPTEN "")
set(OpenFHE_ARCHITECTURE "arm64")
set(OpenFHE_BACKEND_FLAGS_BASE "-DMATHBACKEND=4")

# Compile Definitions
if("ON")
    set(OpenFHE_BINFHE_COMPILE_DEFINITIONS "_compile_defs-NOTFOUND")
    set(OpenFHE_CORE_COMPILE_DEFINITIONS "_compile_defs-NOTFOUND")
    set(OpenFHE_PKE_COMPILE_DEFINITIONS "_compile_defs-NOTFOUND")
    set(OpenFHE_COMPILE_DEFINITIONS
        ${OpenFHE_BINFHE_COMPILE_DEFINITIONS}
        ${OpenFHE_CORE_COMPILE_DEFINITIONS}
        ${OpenFHE_PKE_COMPILE_DEFINITIONS})
endif()

if("OFF")
    set(OpenFHE_BINFHE_COMPILE_DEFINITIONS_STATIC "")
    set(OpenFHE_CORE_COMPILE_DEFINITIONS_STATIC "")
    set(OpenFHE_PKE_COMPILE_DEFINITIONS_STATIC "")
    set(OpenFHE_COMPILE_DEFINITIONS_STATIC
        ${OpenFHE_BINFHE_COMPILE_DEFINITIONS_STATIC}
        ${OpenFHE_CORE_COMPILE_DEFINITIONS_STATIC}
        ${OpenFHE_PKE_COMPILE_DEFINITIONS_STATIC})
endif()
