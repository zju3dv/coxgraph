cmake_minimum_required(VERSION 2.8.3)
project(agast)

find_package(catkin_simple REQUIRED)
catkin_simple()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # Disable -Werror on clang.
  add_definitions(-Wextra -Wall -pedantic -DHAVE_OPENCV -std=c++11)
else()
  add_definitions(-Wextra -Wall -Werror -pedantic -DHAVE_OPENCV -std=c++0x)
endif()

file(GLOB AGAST_SOURCE_FILES  "${PROJECT_SOURCE_DIR}/src/*.cc")
file(GLOB AGAST_HEADER_FILES  "${PROJECT_SOURCE_DIR}/include/agast/*.h")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

cs_add_library(${PROJECT_NAME} ${AGAST_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

cs_export(CFG_EXTRAS export_flags.cmake)
cs_install()
