cmake_minimum_required(VERSION 2.8.3)
project(vins_estimator)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

find_package(OpenCV REQUIRED)

include_directories(${catkin_INCLUDE_DIRS})
include_directories( ${OpenCV_INCLUDE_DIRS} )


cs_add_library(vins_estimator_lib
    src/parameters.cpp
    src/estimator.cpp
    src/feature_manager.cpp
    src/factor/pose_local_parameterization.cpp
    src/factor/projection_factor.cpp
    src/factor/projection_td_factor.cpp
    src/factor/marginalization_factor.cpp
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
)

target_link_libraries(vins_estimator_lib
  ${catkin_LIBRARIES} 
  ${OpenCV_LIBS}
) 

cs_add_executable(vins_estimator
   src/estimator_node.cpp
)

target_link_libraries(vins_estimator
  vins_estimator_lib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

cs_install()
cs_export()
