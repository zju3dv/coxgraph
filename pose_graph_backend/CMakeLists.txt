cmake_minimum_required(VERSION 2.8.3)
project(pose_graph_backend)

SET(CMAKE_BUILD_TYPE RelWithDebInfo)
# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse4.1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")

include_directories(${catkin_INCLUDE_DIRS})
include_directories("include")
include_directories("thirdparty/DBoW2")
include_directories("thirdparty/okvis")

# Thirdparty files
set(thrdprtysrc
  thirdparty/DBoW2/DBoW2/BowVector.cpp
  thirdparty/DBoW2/DBoW2/FBRISK.cpp
  thirdparty/DBoW2/DBoW2/FeatureVector.cpp
  thirdparty/DBoW2/DBoW2/FORB.cpp
  thirdparty/DBoW2/DBoW2/ScoringObject.cpp
  thirdparty/DBoW2/DUtils/Random.cpp
  thirdparty/DBoW2/DUtils/Timestamp.cpp
  thirdparty/okvis/matcher/DenseMatcher.cpp
  thirdparty/okvis/matcher/MatchingAlgorithm.cpp
  thirdparty/okvis/matcher/ThreadPool.cpp
)

# The actual pose graph backend source files
set(SRC
  src/pose_graph_backend/keyframe.cpp
  src/pose_graph_backend/system.cpp
  src/pose_graph_backend/keyframe-database.cpp
  src/pose_graph_backend/map.cpp
  src/pose_graph_backend/loop-detection.cpp
  src/pose_graph_backend/image-matching-algorithm.cpp
  src/pose_graph_backend/optimizer.cpp
)

cs_add_library(pose_graph_backend_lib
  src/subscriber.cpp
  src/publisher.cpp
  src/parameter-reader.cpp
  src/CameraPoseVisualization.cpp
  ${thrdprtysrc}
  ${SRC}
)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(pose_graph_backend_lib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

cs_add_executable(pose_graph_node
  src/pose_graph_node.cpp
)

target_link_libraries(pose_graph_node
  ${catkin_LIBRARIES}
  pose_graph_backend_lib
)

cs_install()
cs_export()
  
