cmake_minimum_required(VERSION 2.8.3)
project(doxygen_catkin)

find_package(catkin REQUIRED)
catkin_package(CFG_EXTRAS doxygen_catkin.cmake)
catkin_python_setup()

install(FILES ${CATKIN_DEVEL_PREFIX}/bin/doxygen-catkin-filegen
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION})
