cmake_minimum_required(VERSION 2.8.3)
project(pose_graph)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mssse3")

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

find_package(OpenCV REQUIRED)

include_directories(${catkin_INCLUDE_DIRS})
include_directories( ${OpenCV_INCLUDE_DIRS} )

cs_add_library(pose_graph_lib
  src/pose_graph.cpp
  src/keyframe.cpp
  src/utility/CameraPoseVisualization.cpp
  src/ThirdParty/DBoW/BowVector.cpp
  src/ThirdParty/DBoW/FBrief.cpp
  src/ThirdParty/DBoW/FeatureVector.cpp
  src/ThirdParty/DBoW/QueryResults.cpp
  src/ThirdParty/DBoW/ScoringObject.cpp
  src/ThirdParty/DUtils/Random.cpp
  src/ThirdParty/DUtils/Timestamp.cpp
  src/ThirdParty/DVision/BRIEF.cpp
  src/ThirdParty/VocabularyBinary.cpp
)

target_link_libraries(pose_graph_lib
  ${catkin_LIBRARIES}  
  ${OpenCV_LIBS}
) 

cs_add_executable(pose_graph
  src/pose_graph_node.cpp
)

target_link_libraries(pose_graph
  pose_graph_lib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

cs_install()
cs_export()
