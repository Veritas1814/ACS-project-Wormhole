cmake_minimum_required(VERSION 3.15)

# Для macOS: встановлюємо SDK і стандартну бібліотеку до виклику project()
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

project(wormhole CXX)
set(CMAKE_CXX_STANDARD 20)

if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isysroot ${CMAKE_OSX_SYSROOT} -stdlib=libc++ -I${CMAKE_OSX_SYSROOT}/usr/include/c++/v1")
endif()

# Знаходимо залежності
find_package(benchmark REQUIRED)
find_package(nlohmann_json REQUIRED)

# Вказуємо директорії заголовочних файлів (якщо потрібно)
include_directories(${CMAKE_SOURCE_DIR}/include)

# Цілі для тестів (як приклад)
add_executable(test_tree
        src/test_tree.cpp
        src/decision_tree.cpp
        src/node.cpp
        src/decision_forest.cpp
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

# Ціль для бенчмарку (benchmark_wormhole)
add_executable(benchmark_wormhole
        src/benchmark.cpp
        src/decision_tree.cpp
        src/node.cpp
)

# Ціль для plot_result (окрема виконувана програма, що використовує той же код)
add_executable(plot_result
        src/plot_result.cpp
        src/decision_tree.cpp
        src/node.cpp
)

# Оптимізовані версії дерева (як статичні бібліотеки, розташовані в src/modified/)
add_library(test_tree_op1 STATIC src/modified/decision_tree_op1.cpp)
add_library(test_tree_op2 STATIC src/modified/decision_tree_op2.cpp)
add_library(test_tree_op3 STATIC src/modified/decision_tree_op3.cpp)
add_library(test_tree_op4 STATIC src/modified/decision_tree_op4.cpp)
add_library(test_tree_final STATIC src/modified/decision_tree_final.cpp)

target_link_libraries(test_tree_op1 PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_tree_op2 PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_tree_op3 PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_tree_op4 PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_tree_final PRIVATE nlohmann_json::nlohmann_json)

target_link_libraries(test_tree PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_forest PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_forest_tp PRIVATE nlohmann_json::nlohmann_json)


# Лінкування бібліотек до цілей
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