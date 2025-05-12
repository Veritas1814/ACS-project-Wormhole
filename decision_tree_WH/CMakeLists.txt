add_executable(forest_example
  forest/forest.cpp
  forest/src/decision_forest.cpp
  forest/src/decision_tree_final.cpp
#   forest/kernels/compute/forest_compute.cpp
#   forest/kernels/dataflow/forest_reader.cpp
#   forest/kernels/dataflow/forest_writer.cpp
)

target_include_directories(forest_example PRIVATE 
    ${CMAKE_SOURCE_DIR}/decision_tree_WH/forest/include
)

target_link_libraries(forest_example PRIVATE Metalium::Metal)