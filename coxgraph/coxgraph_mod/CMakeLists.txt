cmake_minimum_required(VERSION 2.8.3)
project(coxgraph_mod)

add_definitions(-std=c++14 -Werror)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

cs_add_library(vio_interface
  src/vio_interface.cpp)
target_link_libraries(vio_interface ${OpenCV_LIBS})

##########
# EXPORT #
##########

cs_install()
cs_export()
