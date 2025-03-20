cmake_minimum_required(VERSION 3.15)
project(wormhole CXX)
set(CMAKE_CXX_STANDARD 20)
add_executable(test_tree
decision_tree.cpp)
add_executable(test_forest
decision_forest.cpp)

find_package(nlohmann_json REQUIRED)
target_link_libraries(test_tree nlohmann_json::nlohmann_json)
target_link_libraries(test_forest nlohmann_json::nlohmann_json)