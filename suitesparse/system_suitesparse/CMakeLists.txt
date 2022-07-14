cmake_minimum_required(VERSION 2.8.3)
project(system_suitesparse)

find_package(catkin REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${system_suitesparse_SOURCE_DIR}/cmake")

find_package(SUITESPARSE REQUIRED)

if(NOT SUITESPARSE_FOUND)
  message(FATAL Unable to find suitesparse! Is it not installed on the system? On Ubuntu use `sudo apt install libsuitesparse-dev`.)
endif()

catkin_package(INCLUDE_DIRS ${SUITESPARSE_INCLUDE_DIRS}
          LIBRARIES ${SUITESPARSE_LIBRARIES})
