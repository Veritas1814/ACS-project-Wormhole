# Some flags for tt-metal -- ccache + export compile commands for IDE if needed
set(CMAKE_DISABLE_PRECOMPILE_HEADERS TRUE)
set(ENABLE_CCACHE TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(TT_UNITY_BUILDS OFF)
set(CPM_SOURCE_CACHE "${CMAKE_SOURCE_DIR}/third-party/tt-metal/.cpmcache" CACHE STRING "Path to CPM source cache")

# Save previous CMake CXX flags
set(_PREV_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-dangling-reference")
endif()

add_subdirectory(tt-metal)

# Restore previous CMake CXX flags
set(CMAKE_CXX_FLAGS "${_PREV_CMAKE_CXX_FLAGS}")
unset(_PREV_CMAKE_CXX_FLAGS)


# Now wrap all the headers and .so files nicely into one target

add_library(_tt-metal SHARED IMPORTED GLOBAL)
set_target_properties(_tt-metal PROPERTIES
    IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/third-party/tt-metal/tt_metal/libtt_metal.so"
)

add_library(_ttnn SHARED IMPORTED GLOBAL)
set_target_properties(_ttnn PROPERTIES
    IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/third-party/tt-metal/ttnn/_ttnn.so"
)

add_library(_device SHARED IMPORTED GLOBAL)
set_target_properties(_device PROPERTIES
    IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/third-party/tt-metal/tt_metal/third_party/umd/device/libdevice.so"
)

# Now create a logical interface library
add_library(tt-metal INTERFACE)

# Link the .so files into it
target_link_libraries(tt-metal INTERFACE _tt-metal _ttnn _device)

# Set architecture
if ("$ENV{ARCH_NAME}" STREQUAL "grayskull")
  set(ARCH_NAME "grayskull")
  set(ARCH_EXTRA_DIR "grayskull")
elseif ("$ENV{ARCH_NAME}" STREQUAL "wormhole_b0")
  set(ARCH_NAME "wormhole")
  set(ARCH_EXTRA_DIR "wormhole/wormhole_b0_defines")
elseif ("$ENV{ARCH_NAME}" STREQUAL "blackhole")
  set(ARCH_NAME "blackhole")
  set(ARCH_EXTRA_DIR "blackhole")
else()
  message(FATAL_ERROR "Unsupported ARCH_NAME: $ENV{ARCH_NAME}")
endif()

# Now all the incudes
target_include_directories(tt-metal INTERFACE
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
    ${CMAKE_SOURCE_DIR}/third-party/tt-metal/tt_metal/hw/inc/${ARCH_NAME}
    ${CMAKE_SOURCE_DIR}/third-party/tt-metal/tt_metal/hw/inc/${ARCH_EXTRA_DIR}
    ${CMAKE_SOURCE_DIR}/third-party/tt-metal/tt_metal/third_party/umd/src/firmware/riscv/${ARCH_NAME}
    # ${CMAKE_SOURCE_DIR}/third-party/tt-metal/tt_metal/third_party/tracy/public
    # ${CMAKE_SOURCE_DIR}/third-party/tt-metal/tt_metal/third_party/taskflow
    ${CMAKE_SOURCE_DIR}/third-party/tt-metal/tt_stl
    ${CMAKE_SOURCE_DIR}/third-party/tt-metal/tt_stl/tt_stl
    # ${CMAKE_SOURCE_DIR}/third-party/tt-metal/tt_eager
    # ${CMAKE_SOURCE_DIR}/third-party/tt-metal-build/include
    ${CMAKE_SOURCE_DIR}/third-party/tt-metal/.cpmcache/reflect/e75434c4c5f669e4a74e4d84e0a30d7249c1e66f
    ${CMAKE_SOURCE_DIR}/third-party/tt-metal/.cpmcache/nanomsg/28cc32d5bdb6a858fe53b3ccf7e923957e53eada/include
    ${CMAKE_SOURCE_DIR}/third-party/tt-metal/.cpmcache/fmt/69912fb6b71fcb1f7e5deca191a2bb4748c4e7b6//include
    ${CMAKE_SOURCE_DIR}/third-party/tt-metal/.cpmcache/magic_enum/4d76fe0a5b27a0e62d6c15976d02b33c54207096/include
    ${CMAKE_SOURCE_DIR}/third-party/tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/core/include
    # ${CMAKE_SOURCE_DIR}/third-party/tt-metal/.cpmcache/nlohmann_json/798e0374658476027d9723eeb67a262d0f3c8308/include
    ${CMAKE_SOURCE_DIR}/third-party/tt-metal/.cpmcache/xtensor/4a957e26c765b48cbec4a4235fe9e518d5a85d3d/include
    ${CMAKE_SOURCE_DIR}/third-party/tt-metal/.cpmcache/xtensor-blas/190c3a4314355b67291a7d78b20a2100de3f8f54/include
    ${CMAKE_SOURCE_DIR}/third-party/tt-metal/.cpmcache/xtl/0918808959d33a292c551b9f014a0e808bc4a95c/include
)
