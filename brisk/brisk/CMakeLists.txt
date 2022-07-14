cmake_minimum_required(VERSION 2.8)
include(CheckIncludeFileCXX)

project(brisk)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

include_directories(${CATKIN_DEVEL_PREFIX}/include)

add_definitions(-Wextra -Wall -pedantic)

if (NOT (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  add_definitions( -Werror )
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse4.1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")

cs_add_library(${PROJECT_NAME} src/brisk-descriptor-extractor.cc
                               src/brisk-feature-detector.cc
                               src/brisk-layer.cc
                               src/brisk-opencv.cc
                               src/brisk-scale-space.cc
                               src/brute-force-matcher.cc
                               src/harris-feature-detector.cc
                               src/harris-score-calculator.cc
                               src/harris-score-calculator-float.cc
                               src/harris-scores.cc
                               src/image-down-sampling.cc
                               src/pattern-provider.cc
                               src/vectorized-filters.cc
                               src/test/image-io.cc
                               src/timer.cc)

if (IS_SSE_ENABLED)
  cs_add_library(${PROJECT_NAME}_sse src/camera-aware-feature.cc
                                 src/brisk-v1.cc)

  cs_add_executable(camera src/test-cameras.cc)
  target_link_libraries(camera ${PROJECT_NAME} ${PROJECT_NAME}_sse)
else()
  message(WARNING "Neon instructions not implemented for ethzasl_brisk.")
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})


# Testing.
##########
enable_testing()
file(COPY src/test/test_data DESTINATION ${PROJECT_BINARY_DIR})

cs_add_library(${PROJECT_NAME}_test_lib src/test/serialization.cc
                                        src/test/bench-ds.cc
                                        src/opencv-ref.cc)
target_link_libraries(${PROJECT_NAME}_test_lib ${PROJECT_NAME})

catkin_add_gtest(test_integral_image src/test/test-integral-image.cc
                 WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
target_link_libraries(test_integral_image ${GLOG_LIBRARY}
                                          ${PROJECT_NAME}
                                          ${PROJECT_NAME}_test_lib)

catkin_add_gtest(test_popcnt src/test/test-popcount.cc
                 WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
target_link_libraries(test_popcnt ${GLOG_LIBRARY}
                                    ${PROJECT_NAME}
                                    ${PROJECT_NAME}_test_lib)

catkin_add_gtest(test_downsampling src/test/test-downsampling.cc
                 WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
target_link_libraries(test_downsampling ${GLOG_LIBRARY}
                                        ${PROJECT_NAME}
                                        ${PROJECT_NAME}_test_lib)

# TODO(slynen): The test files from Linux don't verify binary equal on OSX.
# TODO(fabianbl): Same for ARM processors.
if(NOT APPLE AND NOT IS_NEON_ENABLED)
  catkin_add_gtest(test_binary_equal src/test/test-binary-equal.cc
                   WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  target_link_libraries(test_binary_equal ${GLOG_LIBRARY}
                                          ${PROJECT_NAME}
                                          ${PROJECT_NAME}_test_lib)
endif()

catkin_add_gtest(test_match src/test/test-match.cc
                 WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
target_link_libraries(test_match ${GLOG_LIBRARY}
                                   ${PROJECT_NAME}
                                   ${PROJECT_NAME}_test_lib)

catkin_add_gtest(test_serialization src/test/test-serialization.cc
                 WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
target_link_libraries(test_serialization ${GLOG_LIBRARY}
                                         ${PROJECT_NAME}
                                         ${PROJECT_NAME}_test_lib)

cs_export()
cs_install()
