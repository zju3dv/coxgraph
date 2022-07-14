cmake_minimum_required(VERSION 2.8.3)
project(opencv3_catkin)

find_package(catkin_simple REQUIRED)

catkin_simple()

find_package(OpenCV COMPONENTS aruco calib3d core features2d flann highgui imgcodecs imgproc line_descriptor ml objdetect phase_unwrapping photo rgbd saliency shape stitching superres video videoio videostab viz xfeatures2d ximgproc REQUIRED )

cs_install()
cs_export(INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS}
          LIBRARIES ${OpenCV_LIBS})
