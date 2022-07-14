cmake_minimum_required(VERSION 2.4.6)
project(brisk_ros_demo)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

add_definitions(-march=native -Wextra -Wall
                -Wno-overloaded-virtual -Wno-non-virtual-dtor # OpenCV.
                -std=c++0x -DENABLE_BRISK_TIMING -DTEST_IN_SOURCE
                -DHAVE_OPENCV)

if (NOT (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  add_definitions( -Werror )
endif()

cs_add_executable(livedemo src/livedemo.cc)
target_link_libraries(livedemo ${catkin_LIBRARIES})
