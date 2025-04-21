# TT-Metal dependency
if (BUILD_WORMHOLE_DECISION_TREE)
  # Save previous CMake CXX flags
  set(_PREV_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-dangling-reference") # Required to build with gcc
  endif()

  # Some flags for tt-metal -- ccache + unity build + make sure that cpm cache stays in tt-metal dir
  # set(CMAKE_DISABLE_PRECOMPILE_HEADERS TRUE)
  # set(ENABLE_CCACHE TRUE)
  # set(TT_UNITY_BUILDS OFF)  # Unity builds for some reason force recompilation every time
  set(CPM_SOURCE_CACHE "${CMAKE_SOURCE_DIR}/third-party/tt-metal/.cpmcache" CACHE STRING "Path to CPM source cache")

  add_subdirectory(tt-metal)

  # Restore previous CMake CXX flags
  set(CMAKE_CXX_FLAGS "${_PREV_CMAKE_CXX_FLAGS}")
  unset(_PREV_CMAKE_CXX_FLAGS)

  set(METALIUM_INCLUDE_DIRS
      ${CMAKE_SOURCE_DIR}/third-party/tt-metal/ttnn
      ${CMAKE_SOURCE_DIR}/third-party/tt-metal/ttnn/cpp
      ${CMAKE_SOURCE_DIR}/third-party/tt-metal/ttnn/cpp/ttnn/deprecated
      ${CMAKE_SOURCE_DIR}/third-party/tt-metal/tt_metal/api
      ${CMAKE_SOURCE_DIR}/third-party/tt-metal/tt_metal
      ${CMAKE_SOURCE_DIR}/third-party/tt-metal/tt_metal/include
      ${CMAKE_SOURCE_DIR}/third-party/tt-metal/tt_metal/hostdevcommon/api
      ${CMAKE_SOURCE_DIR}/third-party/tt-metal/tt_metal/third_party/umd
      ${CMAKE_SOURCE_DIR}/third-party/tt-metal/tt_metal/third_party/umd/device/api
      ${CMAKE_SOURCE_DIR}/third-party/tt-metal/tt_metal/hw/inc
      ${CMAKE_SOURCE_DIR}/third-party/tt-metal/tt_stl
      ${CMAKE_SOURCE_DIR}/third-party/tt-metal/tt_stl/tt_stl
  )

  # Now wrap all the headers and .so files nicely into one target
  if(NOT TARGET Metalium::Metal)
    set(METALIUM_LIB_PATH "${CMAKE_BINARY_DIR}/lib")
    find_library(TT_METAL_LIBRARY NAMES "tt_metal" PATHS "${METALIUM_LIB_PATH}" NO_DEFAULT_PATH)
    find_library(DEVICE_LIBRARY NAMES "device" PATHS "${METALIUM_LIB_PATH}" NO_DEFAULT_PATH)

    if(TT_METAL_LIBRARY)
      add_library(Metalium::Metal SHARED IMPORTED GLOBAL)
      set_target_properties(Metalium::Metal PROPERTIES
          IMPORTED_LOCATION "${TT_METAL_LIBRARY}"
          INTERFACE_INCLUDE_DIRECTORIES "${METALIUM_INCLUDE_DIRS}"
      )
      target_link_libraries(
        Metalium::Metal
        INTERFACE
            ${DEVICE_LIBRARY}
            nlohmann_json::nlohmann_json
        )
        message(STATUS "Successfully found libtt_metal.so at ${TT_METAL_LIBRARY}")
      else()
          message(FATAL_ERROR "libtt_metal.so not found in ${METALIUM_LIB_PATH}")
      endif()
  else()
      message(STATUS "Metalium targets already exists")
  endif()
endif()