cmake_minimum_required(VERSION 3.15)

# For macOS: install SDK and common library to call project()
if(APPLE)
    execute_process(
            COMMAND xcrun --sdk macosx --show-sdk-path
            OUTPUT_VARIABLE SDK_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(CMAKE_OSX_SYSROOT ${SDK_PATH} CACHE STRING "OSX SDK" FORCE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING "Minimum OS X deployment version" FORCE)
    message(STATUS "Using macOS SDK: ${CMAKE_OSX_SYSROOT}")
endif()

project(wormhole C CXX)
set(CMAKE_CXX_STANDARD 20)

option(BUILD_WORMHOLE_DECISION_TREE "Build wormhole implementation of decision tree" OFF)

if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isysroot ${CMAKE_OSX_SYSROOT} -stdlib=libc++ -I${CMAKE_OSX_SYSROOT}/usr/include/c++/v1")
endif()

# ============ Dependencies via FetchContent ============

include(FetchContent)

set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG main
)


FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG develop
)

FetchContent_MakeAvailable(benchmark nlohmann_json)

# ============ Include Paths ============

include_directories(${CMAKE_SOURCE_DIR}/include)

# ============ Executables ============

add_executable(test_tree
        src/test_tree.cpp
        src/decision_tree.cpp
        src/node.cpp
)

add_executable(test_forest
        src/test_forest.cpp
        src/decision_forest.cpp
        src/node.cpp
        src/decision_tree.cpp
)

add_executable(test_forest_tp
        src/test_forest.cpp
        src/decision_forest_tp.cpp
        src/node.cpp
        src/decision_tree.cpp
        src/thread_pool.cpp
)

add_executable(benchmark_wormhole
        src/benchmark.cpp
        src/decision_tree.cpp
        src/node.cpp
)

add_executable(plot_result
        src/plot_result.cpp
        src/decision_tree.cpp
        src/node.cpp
)

# ============ Static Libraries ============

add_library(test_tree_op1 STATIC src/modified/decision_tree_op1.cpp)
add_library(test_tree_op2 STATIC src/modified/decision_tree_op2.cpp)
add_library(test_tree_op3 STATIC src/modified/decision_tree_op3.cpp)
add_library(test_tree_op4 STATIC src/modified/decision_tree_op4.cpp)
add_library(test_tree_final STATIC src/modified/decision_tree_final.cpp)

# ============ Linking ============

target_link_libraries(test_tree_op1 PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_tree_op2 PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_tree_op3 PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_tree_op4 PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_tree_final PRIVATE nlohmann_json::nlohmann_json)

target_link_libraries(test_tree PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_forest PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_forest_tp PRIVATE nlohmann_json::nlohmann_json)

target_link_libraries(benchmark_wormhole
        PRIVATE benchmark::benchmark
        PRIVATE test_tree_op1
        PRIVATE test_tree_op2
        PRIVATE test_tree_op3
        PRIVATE test_tree_op4
        PRIVATE test_tree_final
        PRIVATE nlohmann_json::nlohmann_json
)

target_link_libraries(plot_result
        PRIVATE benchmark::benchmark
        PRIVATE test_tree_op1
        PRIVATE test_tree_op2
        PRIVATE test_tree_op3
        PRIVATE test_tree_op4
        PRIVATE test_tree_final
        PRIVATE nlohmann_json::nlohmann_json
)

add_subdirectory(third_party)

if (BUILD_WORMHOLE_DECISION_TREE)
        add_subdirectory(decision_tree_WH)
endif()