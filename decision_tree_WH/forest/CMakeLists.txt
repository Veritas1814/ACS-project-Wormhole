cmake_minimum_required(VERSION 4.0)
project(forest_project LANGUAGES CXX)

# ——————————————————————————————————————————————————————————
# 1) Compiler settings
# ——————————————————————————————————————————————————————————
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ——————————————————————————————————————————————————————————
# 2) Locate TT-Metalium (must have installed it or built it,
#    and set CMAKE_PREFIX_PATH to its install prefix)
# ——————————————————————————————————————————————————————————
if(NOT DEFINED TT_METALIUM_ROOT)
  set(TT_METALIUM_ROOT $ENV{TT_METALIUM_ROOT})
endif()
if(TT_METALIUM_ROOT)
  list(APPEND CMAKE_PREFIX_PATH ${TT_METALIUM_ROOT})
endif()

find_package(tt-metalium CONFIG REQUIRED)

# ——————————————————————————————————————————————————————————
# 3) Locate JSON for DecisionTree/RandomForest (if you end up using them)
# ——————————————————————————————————————————————————————————
find_package(nlohmann_json CONFIG REQUIRED)

# ——————————————————————————————————————————————————————————
# 4) Include directories
# ——————————————————————————————————————————————————————————
include_directories(
  ${PROJECT_SOURCE_DIR}/include
)

# ——————————————————————————————————————————————————————————
# 5) Host executable
# ——————————————————————————————————————————————————————————
add_executable(forest_app
  forest.cpp
)

target_include_directories(forest_app PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  # ${PROJECT_SOURCE_DIR}/root/c150661229a53d9c021900f2235cc3a1/tt-metal/tt_metal/programming_examples/forest/include
)

target_link_libraries(forest_app PRIVATE
  tt-metalium::ttm-host       # TT-Metalium host-API library
  nlohmann_json::nlohmann_json
)

# ——————————————————————————————————————————————————————————
# 6) Copy kernel sources into build directory for CreateKernel()
# ——————————————————————————————————————————————————————————
set(KERNEL_SRCS
  ${PROJECT_SOURCE_DIR}/kernels/dataflow/forest_reader.cpp
  ${PROJECT_SOURCE_DIR}/kernels/dataflow/forest_writer.cpp
)

foreach(src IN LISTS KERNEL_SRCS)
  get_filename_component(fn ${src} NAME)
  configure_file(${src} ${fn} COPYONLY)
endforeach()

# ——————————————————————————————————————————————————————————
# 7) Install rules
# ——————————————————————————————————————————————————————————
install(TARGETS forest_app
  RUNTIME DESTINATION bin
)

# Install headers and original kernels (optional)
install(DIRECTORY include/ DESTINATION include)
install(FILES
  ${PROJECT_SOURCE_DIR}/kernels/dataflow/forest_reader.cpp
  ${PROJECT_SOURCE_DIR}/kernels/dataflow/forest_writer.cpp
  DESTINATION kernels/dataflow
)
