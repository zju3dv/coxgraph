cmake_minimum_required(VERSION 2.8.3)
project(feature_tracker)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

find_package(OpenCV REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
include_directories( ${OpenCV_INCLUDE_DIRS} )


cs_add_library(feature_tracker_lib
  src/parameters.cpp
  src/feature_tracker.cpp
)  

target_link_libraries(feature_tracker_lib
  ${catkin_LIBRARIES} 
  ${OpenCV_LIBS}
)

cs_add_executable(feature_tracker
  src/feature_tracker_node.cpp
)

target_link_libraries(feature_tracker
  feature_tracker_lib
  ${catkin_LIBRARIES} 
  ${OpenCV_LIBS}
)


cs_install()
cs_export()
