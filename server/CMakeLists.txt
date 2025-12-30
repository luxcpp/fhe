cmake_minimum_required(VERSION 3.14)
project(fhe_server CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set paths
set(FHE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(HTTP_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../http)

# Find dependencies
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)
find_package(OpenSSL REQUIRED)
find_package(ZLIB REQUIRED)

# Find Brotli
pkg_check_modules(BROTLI libbrotlidec libbrotlienc)

# Find PostgreSQL
find_package(PostgreSQL QUIET)

# Find c-ares
pkg_check_modules(CARES libcares)

# Find SQLite3
find_package(SQLite3 QUIET)

# FHE Server executable
add_executable(${PROJECT_NAME} main.cc)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${FHE_ROOT}/install/include/openfhe
    ${FHE_ROOT}/install/include/openfhe/core
    ${FHE_ROOT}/install/include/openfhe/binfhe
    ${FHE_ROOT}/install/include/openfhe/pke
    ${FHE_ROOT}/src/binfhe/include
    ${FHE_ROOT}/src/core/include  
    ${FHE_ROOT}/src/pke/include
    ${FHE_ROOT}/third-party/cereal/include
    ${HTTP_ROOT}/lib/inc
    ${HTTP_ROOT}/orm_lib/inc
    ${HTTP_ROOT}/nosql_lib/redis/inc
    ${HTTP_ROOT}/trantor
    ${HTTP_ROOT}/build/trantor/exports
    ${HTTP_ROOT}/build/exports
    ${JSONCPP_INCLUDE_DIRS}
)

# Link directories
target_link_directories(${PROJECT_NAME} PRIVATE
    ${FHE_ROOT}/install/lib
    ${HTTP_ROOT}/build
    ${HTTP_ROOT}/build/trantor
    ${JSONCPP_LIBRARY_DIRS}
    ${BROTLI_LIBRARY_DIRS}
    ${CARES_LIBRARY_DIRS}
)

# Link libraries  
target_link_libraries(${PROJECT_NAME} PRIVATE
    http
    trantor
    ${JSONCPP_LIBRARIES}
    OpenSSL::SSL
    OpenSSL::Crypto
    ZLIB::ZLIB
    FHEbinfhe
    FHEcore
    FHEpke
    pthread
)

# Add optional dependencies
if(BROTLI_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${BROTLI_LIBRARIES})
endif()

if(CARES_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CARES_LIBRARIES})
endif()

if(PostgreSQL_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE PostgreSQL::PostgreSQL)
endif()

if(SQLite3_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE SQLite::SQLite3)
endif()

# Find and link MySQL if available
find_path(MYSQL_INCLUDE_DIR mysql/mysql.h PATHS /opt/homebrew/include)
find_library(MYSQL_LIBRARY NAMES mysqlclient mariadb PATHS /opt/homebrew/lib)
if(MYSQL_LIBRARY)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${MYSQL_LIBRARY})
endif()

# macOS specific
if(APPLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        "-framework CoreFoundation"
        "-framework Security"
    )
endif()

# Copy to bin directory
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# RPATH for finding FHE libs
set_target_properties(${PROJECT_NAME} PROPERTIES
    BUILD_RPATH "${FHE_ROOT}/install/lib"
    INSTALL_RPATH "${FHE_ROOT}/install/lib"
)

install(TARGETS ${PROJECT_NAME} DESTINATION bin)

message(STATUS "FHE Server configured")
message(STATUS "  FHE Root: ${FHE_ROOT}")
message(STATUS "  HTTP Root: ${HTTP_ROOT}")
