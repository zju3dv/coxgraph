# Robotic Optimization (ROBOPT) Library #
Package to simplify implementation of optimization problems in Ceres commonly used in robotics.  

This repository is used in a wider framework for multi-robot state estimation and path planning, available [here](https://github.com/VIS4ROB-lab/multi_robot_coordination).  
If you use this code in your academic work, please cite ([PDF](https://www.research-collection.ethz.ch/handle/20.500.11850/441280)):

    @inproceedings{bartolomei2020multi,
      title={Multi-robot Coordination with Agent-Server Architecture for Autonomous Navigation in Partially Unknown Environments},
      author={Bartolomei, Luca and Karrer, Marco and Chli, Margarita},
      booktitle={2020 {IEEE/RSJ} International Conference on Intelligent Robots and Systems ({IROS})},
      year={2020}
    }

## Installation  
In order to install RobOpt Open, follow these steps. First, create a catkin workspace:
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
$ wstool merge robopt_open/dependencies_ssh.rosinstall # To clone with https: robopt_open/dependencies_https.rosinstall
$ wstool up -j8
```  

Finally, build the package:
```
$ cd ~/catkin_ws
$ catkin build robopt_open
```  
