cmake_minimum_required(VERSION 2.8.3)
project(ethzasl_apriltag2)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

add_definitions("-fPIC -O3")
find_package(Threads)

if ("!${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # The clang compiler (on osx) is somehow much more strict
  # than the compilers on ubuntu and so this does not seem
  # possible on OSX just yet.
  add_definitions( -Werror )
endif()

#############
# LIBRARIES #
#############
file(GLOB SOURCE_FILES "src/*.cc")

cs_add_library(${PROJECT_NAME}
  ${SOURCE_FILES}
)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

############
# EXAMPLES #
############
if(NOT APPLE)
  cs_add_executable(apriltags_demo 
    src/example/apriltags_demo.cpp src/example/Serial.cpp
  )
  target_link_libraries(apriltags_demo ${PROJECT_NAME} v4l2)
endif()

##########
# EXPORT #
##########
cs_install()
cs_export()
