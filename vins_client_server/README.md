# VINS-Mono for Client-Server Architecture
This repository contains a client-server version of [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono). It is based on the original VINS-Mono implementation, with the additional capacity of sending information back to a server fusing the sensory data from multiple robots.  
This repository is used in a wider framework for multi-robot path planning, available [here](https://github.com/VIS4ROB-lab/multi_robot_coordination).  

If you use this VINS-Mono version in your academic work, please cite ([PDF](https://www.research-collection.ethz.ch/handle/20.500.11850/441280)):

    @inproceedings{bartolomei2020multi,
      title={Multi-robot Coordination with Agent-Server Architecture for Autonomous Navigation in Partially Unknown Environments},
      author={Bartolomei, Luca and Karrer, Marco and Chli, Margarita},
      booktitle={2020 {IEEE/RSJ} International Conference on Intelligent Robots and Systems ({IROS})},
      year={2020}
    }

## Installation  
In order to install this VINS-Mono version, follow these steps (tested under Ubuntu 18.04 LTS, ROS Melodic).
First, install these dependencies:
```
$ sudo apt-get install ros-melodic-cv-bridge ros-melodic-tf ros-melodic-message-filters ros-melodic-image-transport
```  

Then, create a catkin workspace:
```
$ mkdir -p catkin_ws/src
$ cd catkin_ws
```
Set-up the workspace:
```
$ catkin init
$ catkin config --extend /opt/ros/melodic
$ catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
$ catkin config --merge-devel
```

Clone the dependencies:
```
$ cd ~/catkin_ws/src
$ wstool init
$ wstool merge vins_client_server/dependencies_ssh.rosinstall # To clone with https: vins_client_server/dependencies_https.rosinstall
$ wstool up -j8
```  

Finally, build the package:
```
$ cd ~/catkin_ws
$ catkin build vins_client_server
```  

For usage and setting of the parameters, we refer to the [original VINS-Mono repository](https://github.com/HKUST-Aerial-Robotics/VINS-Mono).
