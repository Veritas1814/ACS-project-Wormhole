cmake_minimum_required(VERSION 3.15)
project(wormhole CXX)

set(CMAKE_CXX_STANDARD 20)

# Find dependencies
find_package(benchmark REQUIRED)
find_package(nlohmann_json REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Executables
add_executable(test_tree src/test_tree.cpp src/decision_tree.cpp src/node.cpp src/decision_forest.cpp)
add_executable(test_forest src/test_forest.cpp src/decision_forest.cpp src/node.cpp src/decision_tree.cpp)
add_executable(benchmark_wormhole src/benchmark.cpp src/decision_tree.cpp src/node.cpp)

# Optimized versions of decision tree
add_library(test_tree_op1 STATIC src/modified/decision_tree_op1.cpp)
add_library(test_tree_op2 STATIC src/modified/decision_tree_op2.cpp)
add_library(test_tree_op3 STATIC src/modified/decision_tree_op3.cpp)
add_library(test_tree_op4 STATIC src/modified/decision_tree_op4.cpp)
add_library(test_tree_op5 STATIC src/modified/decision_tree_op5.cpp)
add_library(test_tree_op6 STATIC src/modified/decision_tree_op6.cpp)


# Link libraries to all executables
target_link_libraries(test_tree nlohmann_json::nlohmann_json)
target_link_libraries(test_forest nlohmann_json::nlohmann_json)
target_link_libraries(benchmark_wormhole
        benchmark::benchmark
        test_tree_op1
        test_tree_op2
        test_tree_op3
        test_tree_op4
        test_tree_op5
        test_tree_op6
        nlohmann_json::nlohmann_json
)