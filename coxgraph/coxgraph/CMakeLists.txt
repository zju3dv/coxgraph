cmake_minimum_required(VERSION 2.8.3)
project(coxgraph)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

add_definitions(-std=c++17 -Werror)

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

find_package(Open3D REQUIRED)

include_directories(${Open3D_INCLUDE_DIRS})
link_directories(${Open3D_LIBRARY_DIRS})

cs_add_library(${PROJECT_NAME}
    src/client/coxgraph_client.cpp
    src/client/map_server.cpp
    src/server/coxgraph_server.cpp
    src/server/client_handler.cpp
    src/server/pose_graph_interface.cpp
    src/server/global_tf_controller.cpp
    src/server/submap_collection.cpp
    src/server/client_tf_optimizer.cpp
    src/server/visualizer/server_visualizer.cpp)
#target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})
message(STATUS ${Open3D_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${Open3D_LIBRARIES})

cs_add_executable(coxgraph_server_node
    src/coxgraph_server_node.cpp)
target_link_libraries(coxgraph_server_node ${PROJECT_NAME})

cs_add_executable(coxgraph_client_node
    src/coxgraph_client_node.cpp)
target_link_libraries(coxgraph_client_node ${PROJECT_NAME})

cs_add_executable(tsdf_recover_node
    src/tsdf_recover_node.cpp)
target_link_libraries(tsdf_recover_node ${PROJECT_NAME})

cs_export()
