cmake_minimum_required(VERSION 2.8.3)
project(pcl_catkin)

find_package(catkin_simple REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem thread)

catkin_simple(ALL_DEPS_REQUIRED)

include(ExternalProject)

set(PCL_CATKIN_INCLUDE ${CATKIN_DEVEL_PREFIX}/include/pcl-1.9)
file(MAKE_DIRECTORY ${PCL_CATKIN_INCLUDE})

if(NOT CMAKE_BUILD_TYPE)
  MESSAGE(FATAL_ERROR "CMAKE_BUILD_TYPE must not be empty!")
endif()
string(TOUPPER ${CMAKE_BUILD_TYPE} UC_BUILD_TYPE)

# PCL uses -march=native which can be dangerous to use together with Eigen 3.3
# as this can introduce different vectorization settings in different parts
# of the code. So we manually set the compiler flags here to avoid this issue.
include("${CMAKE_SOURCE_DIR}/cmake/pcl_find_sse.cmake")
PCL_CHECK_FOR_SSE()

# Flags that PCL defines by default, see https://github.com/PointCloudLibrary/pcl/blob/master/CMakeLists.txt#L112
SET(PCL_CXX_FLAGS "-Wall -Wextra -Wno-unknown-pragmas -fno-strict-aliasing -Wno-format-extra-args -Wno-sign-compare -Wno-invalid-offsetof -Wno-conversion -DBOOST_UUID_RANDOM_GENERATOR_COMPAT ${SSE_FLAGS} -Wabi")
SET(PCL_CXX_FLAGS "${PCL_CXX_FLAGS} -std=c++11")

ExternalProject_Add(pcl_src
  #URL http://192.168.0.2:8000/Libraries/packages/pcl-1.9.1.tar.gz
  GIT_REPOSITORY  git@github.com:PointCloudLibrary/pcl.git
  GIT_TAG pcl-1.9.1
  UPDATE_COMMAND ""
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${CATKIN_DEVEL_PREFIX}
             -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
             -DCMAKE_CXX_FLAGS:STRING=${PCL_CXX_FLAGS}
             -DPCL_SHARED_LIBS=TRUE
             "-DCMAKE_CXX_FLAGS_${UC_BUILD_TYPE}=${CMAKE_CXX_FLAGS_${UC_BUILD_TYPE}} ${PCL_CXX_FLAGS}"
)

cs_add_library(${PROJECT_NAME} src/dependency_tracker.cc)
add_dependencies(${PROJECT_NAME} pcl_src)

install(DIRECTORY ${CATKIN_DEVEL_PREFIX}/lib/
        DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
      FILES_MATCHING PATTERN "libpcl*")

# cs_install()
install(DIRECTORY ${PCL_CATKIN_INCLUDE}
  DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.hpp"
  PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)

cs_export(CFG_EXTRAS pcl-extras.cmake)
