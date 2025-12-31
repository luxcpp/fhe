#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "FHEcore" for configuration "Release"
set_property(TARGET FHEcore APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(FHEcore PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libFHEcore.1.4.2.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libFHEcore.1.dylib"
  )

list(APPEND _cmake_import_check_targets FHEcore )
list(APPEND _cmake_import_check_files_for_FHEcore "${_IMPORT_PREFIX}/lib/libFHEcore.1.4.2.dylib" )

# Import target "FHEpke" for configuration "Release"
set_property(TARGET FHEpke APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(FHEpke PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libFHEpke.1.4.2.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libFHEpke.1.dylib"
  )

list(APPEND _cmake_import_check_targets FHEpke )
list(APPEND _cmake_import_check_files_for_FHEpke "${_IMPORT_PREFIX}/lib/libFHEpke.1.4.2.dylib" )

# Import target "FHEbin" for configuration "Release"
set_property(TARGET FHEbin APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(FHEbin PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libFHEbin.1.4.2.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libFHEbin.1.dylib"
  )

list(APPEND _cmake_import_check_targets FHEbin )
list(APPEND _cmake_import_check_files_for_FHEbin "${_IMPORT_PREFIX}/lib/libFHEbin.1.4.2.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
