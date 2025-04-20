# Inspired by https://github.com/tenstorrent/tt-mlir/blob/b55b6b0fb4149c7bc50d4d467d8fc9964ef082a5/third_party/CMakeLists.txt

if (BUILD_WORMHOLE_DECISION_TREE)
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

  set(TTMETAL_INCLUDE_DIRS
      ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/ttnn
      ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/ttnn/cpp
      ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/deprecated
      ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/api
      ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal
      ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/include
      ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/hostdevcommon/api
      ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/third_party/umd
      ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/third_party/umd/device/api
      ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/hw/inc
      ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/hw/inc/${ARCH_NAME}
      ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/hw/inc/${ARCH_EXTRA_DIR}
      ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_metal/third_party/umd/src/firmware/riscv/${ARCH_NAME}
      ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_stl
      ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/tt_stl/tt_stl
      ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/.cpmcache/reflect/e75434c4c5f669e4a74e4d84e0a30d7249c1e66f
      ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/.cpmcache/nanomsg/28cc32d5bdb6a858fe53b3ccf7e923957e53eada/include
      ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/.cpmcache/fmt/69912fb6b71fcb1f7e5deca191a2bb4748c4e7b6//include
      ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/.cpmcache/magic_enum/4d76fe0a5b27a0e62d6c15976d02b33c54207096/include
      ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/.cpmcache/nlohmann_json/798e0374658476027d9723eeb67a262d0f3c8308/include
      ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/core/include
      ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/.cpmcache/xtensor/4a957e26c765b48cbec4a4235fe9e518d5a85d3d/include
      ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/.cpmcache/xtensor-blas/190c3a4314355b67291a7d78b20a2100de3f8f54/include
      ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal/.cpmcache/xtl/0918808959d33a292c551b9f014a0e808bc4a95c/include
      # PARENT_SCOPE
  )


  set(TT_METAL_VERSION "v0.57.0-rc69")
  set(CMAKE_INSTALL_MESSAGE LAZY)  # suppress "Up-to-date:..." messages in incremental builds

  set(METAL_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal-build/")

  set(TTMETAL_LIBRARY_DIR ${PROJECT_SOURCE_DIR}/third_party/tt-metal/src/tt-metal-build/lib)
  set(TTNN_LIBRARY_PATH ${TTMETAL_LIBRARY_DIR}/_ttnn.so)
  set(TTMETAL_LIBRARY_PATH ${TTMETAL_LIBRARY_DIR}/libtt_metal.so)
  set(DEVICE_LIBRARY_PATH ${TTMETAL_LIBRARY_DIR}/libdevice.so)

  set(TTMETAL_LIBRARY_DIR ${TTMETAL_LIBRARY_DIR} PARENT_SCOPE)
  set(TTNN_LIBRARY_PATH ${TTNN_LIBRARY_PATH} PARENT_SCOPE)
  set(TTMETAL_LIBRARY_PATH ${TTMETAL_LIBRARY_PATH} PARENT_SCOPE)
  set(DEVICE_LIBRARY_PATH ${DEVICE_LIBRARY_PATH} PARENT_SCOPE)


  include(ExternalProject)
  ExternalProject_Add(
    tt-metal
    PREFIX ${PROJECT_SOURCE_DIR}/third_party/tt-metal
    CMAKE_ARGS
      -DCMAKE_INSTALL_MESSAGE=LAZY
    UPDATE_COMMAND
      git submodule update --init --recursive &&
      git submodule foreach "git lfs fetch --all && git lfs pull"
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX=${METAL_INSTALL_PREFIX}
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
      -DCMAKE_CXX_FLAGS="-Wno-dangling-reference" # Required to build with GCC
      -DENABLE_CCACHE=ON
      -DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON
      -DENABLE_TRACY=OFF
      -DENABLE_LIBCXX=OFF
    GIT_REPOSITORY https://github.com/tenstorrent/tt-metal.git
    GIT_TAG ${TT_METAL_VERSION}
    GIT_PROGRESS ON
    BUILD_BYPRODUCTS ${TTNN_LIBRARY_PATH} ${TTMETAL_LIBRARY_PATH} ${DEVICE_LIBRARY_PATH}
  )

  ExternalProject_Add_StepTargets(tt-metal download configure)

  list(APPEND library_names TTNN_LIBRARY TTMETAL_LIBRARY DEVICE_LIBRARY)
  list(APPEND library_paths ${TTNN_LIBRARY_PATH} ${TTMETAL_LIBRARY_PATH} ${DEVICE_LIBRARY_PATH})

  foreach(lib_name lib_path IN ZIP_LISTS library_names library_paths)
    add_library(${lib_name} SHARED IMPORTED GLOBAL)
    set_target_properties(${lib_name} PROPERTIES EXCLUDE_FROM_ALL TRUE IMPORTED_LOCATION ${lib_path})
    add_dependencies(${lib_name} tt-metal)
  endforeach()

  # Now create a logical interface library
  add_library(tt-metal-full INTERFACE)

  # Link the .so files into it
  target_link_libraries(tt-metal-full INTERFACE TTNN_LIBRARY TTMETAL_LIBRARY DEVICE_LIBRARY)

  # Now all the incudes
  target_include_directories(tt-metal-full INTERFACE "${TTMETAL_INCLUDE_DIRS}")
endif()