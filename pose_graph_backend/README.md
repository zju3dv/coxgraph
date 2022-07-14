Pose Graph Backend
==================
This repository contains an optimization-based pose-graph backend for multi-agent sensor fusion. This repository is used in a wider framework for multi-robot path planning, available [here](https://github.com/VIS4ROB-lab/multi_robot_coordination).  

If you use this code in your academic work, please cite ([PDF](https://www.research-collection.ethz.ch/handle/20.500.11850/441280)):

    @inproceedings{bartolomei2020multi,
      title={Multi-robot Coordination with Agent-Server Architecture for Autonomous Navigation in Partially Unknown Environments},
      author={Bartolomei, Luca and Karrer, Marco and Chli, Margarita},
      booktitle={2020 {IEEE/RSJ} International Conference on Intelligent Robots and Systems ({IROS})},
      year={2020}
    }

This project is released under a GPLv3 license.

## Overview ##
This sensor fusion framework is built as part of a Collaborative SLAM client-server framework consisting of the following software packages:
* Client's side (Onboard the UAV):
  * `vins_client_server` - [link](https://github.com/VIS4ROB-lab/vins_client_server)
  * `image_undistort` - [link](https://github.com/ethz-asl/image_undistort)
  * `pcl_fusion` - [link](https://github.com/VIS4ROB-lab/pcl_fusion)
  
* Server's (Backend PC):
  * `pose_graph_backend` - this repository
  * `Multi-agent Voxblox` - [link](https://github.com/VIS4ROB-lab/voxblox_multi_agent)
  * `comm_msgs` - [link](https://github.com/VIS4ROB-lab/comm_msgs)
  
All the above must be installed to test the full system. This was developed and verified in ROS Melodic on Ubuntu 18.04 LTS.

## Installation ##
This software has been tested under Ubuntu 18.04 LTS and ROS Melodic. Here we assume that ROS has been properly installed.  

First, install these dependencies:
```
$ sudo apt-get install python-catkin-tools python-wstool libeigen3-dev
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
$ wstool merge pose_graph_backend/dependencies_ssh.rosinstall # To clone with https: pose_graph_backend/dependencies_https.rosinstall
$ wstool up -j8
```  

Finally, build the workspace:
```
$ cd ~/catkin_ws
$ catkin build
```  

Once the build is complete, source the workspace:
```
$ source devel/setup.bash
```

## Topics ##
The following are the ROS topics the Pose Graph backend node subscribes to. The `X` represents the x-th agent:
* `/keyframeX` - custom keyframe message described in `comm_msgs` sent by `pose graph` node in `vins_client_server`.
* `/odometryX` - current odometry estimate sent by `vins_client_server`.
* `/gpsX` - GPS measurement directly from sensor.
* `/fused_pclX` - custom pointcloud message described in comm_msgs sent by pcl fusion node.
* `/fake_gps2_X` - Leica laser measurement data used in EuRoC dataset to simulate GPS messages.

The following are published by Pose Graph backend:
* `/pcl_transformX` - pointcloud and its world frame pose, described in `comm_msgs`.
* `/pathX` - RVIZ trajectory estimate.
* `/camera_pose_visualX` - RVIZ camera visualization.

## Parameters ##
Parameter | Description
------------ | -------------
gps_align_num_corr | number of required GPS-odometry correspondences to start initial GPS alignment.
gps_align_cov_max | the threshold that the covariance of the initial GPS alignment must be under for acceptance. Increase this value if there is an issue with GPS initialization (system requires this for active gps agents).
rel_pose_corr_min | minimum required number of correspondences following a loop detection relative pose optimization needed to accept the loop closure.
rel_pose_outlier_norm_min | norm residual value that a residual of the loop detection relative pose optimization must be above in order to consider that residual an "outlier" and remove it from the following optimization.
local_opt_window_size | size of the window of recent keyframes for sliding window local Pose Graph optimization.
loop_image_min_matches | minimum keypoint matches in an image to continue with loop closure candidate.
loop_detect_sac_thresh | threshold for classifying a point as an inlier in RANSAC 3D-2D P3P in loop detection. Lower value is a stricter condition.
loop_detect_sac_max_iter | max number of iterations for the RANSAC 3D-2D P3P in loop detection.
loop_detect_min_sac_inliers | minimum number of inliers of RANSAC 3D-2D P3P in loop detection to continue with loop closure candidate.
loop_detect_min_sac_inv_inliers | minimum number of inliers of the INVERSE RANSAC 3D-2D P3P in loop detection to continue with loop closure candidate.
loop_detect_min_pose_inliers | the final threshold of inliers from the relative Pose Graph optimization of the loop detection process to verify loop closure.
loop_detect_reset_time | the amount of time following a loop closure where no new loop closures are looked for for that agent.
loop_detect_skip_kf | number of keyframes to skip loop detection for. (1 processes every keyframe, 2 would process every other keyframe, etc.).
information_XXX | information values for the different residuals in Pose Graph optimization. Higher values indicate more certainty about the measurement.
ignore_gps_altitude | set to TRUE typically if running in a live setup due to large fluctuations in GPS altitude. Uses the odometry altitude value instead.
gps_active_X | whether agent X has an active GPS.
gps_referenceX | the reference point for local GPS coordinates.
gps_offsetX | the translational position offset between IMU and GPS antenna onboard UAV.

## General setup ##
The following describes how to run the system.  
A series of configuration files are necessary to run different components:
* A VINS-Mono `yaml` configuration file is required for the `vins_client_server` [package](https://github.com/VIS4ROB-lab/vins_client_server).
* A stereo `yaml` configuration file is required for the `dense_stereo package` from [image_undistort](https://github.com/ethz-asl/image_undistort) to generate a dense point cloud from a pair of stereo images.
* An `ncamera` yaml configuration file is required for the `pcl_fusion` [package](https://github.com/VIS4ROB-lab/pcl_fusion). 
Examples of these configuration files can be found in the folder `pose_graph_backend/conf`.

## Usage examples ##
### 1. EuRoC Dataset ###
#### 1.1 Single Agent ####
The following instructions shows how to run a single agent of the [EuRoC dataset]((https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)) with a simulation of the full pipeline (using the Leica position measurements as a fake GPS).

First, launch the backend software in separate terminals:
```
$ roslaunch pose_graph_backend pose_graph_node_euroc.launch num_agents:=1
$ roslaunch voxblox_ros euroc_dataset.launch
```
Wait for the messages `[PGB] Sucessfully read the parameters and the vocabulary` and `[PGB] Set callbacks` to be printed in the terminal. This indicates that the pose-graph is ready. Note that the Voxblox node outputs visuals in RVIZ every 2 seconds.

Launch the client software in separate terminals (make sure the `agent_id` parameters match in the launch files):
```
$ roslaunch vins_estimator multi_euroc_0.launch
$ roslaunch pcl_fusion pcl_fusion_node_euroc.launch
```

Finally, launch the EuRoC rosbag specifying the right path:
```
$ roslaunch pose_graph_backend single_agent_play.launch  path:=/path/to/euroc/bagfile.bag
```

#### 1.2 Multi Agent ####
Running the full pipeline with more than one agent is not recommended on a single PC unless you have a powerful PC (since it would be running visual-inertial odometry, dense stereo pointcloud construction and pointcloud filtering for every agent, as well as Pose Graph optimization, loop detection, and mesh reconstruction). A better alternative is to run the client-side software and record its output from the EuRoC data in a bag file for different EuRoC trajectories individually, and then play those bag files simultaneously while running the backend packages. This more closely reflects the actual load on your pc.  

An example of the client software output from the first three EuRoC datasets can be found in the folder `pose_graph_backend/data` as prerecorded bag files. To simulate the multi-agent system follow these instructions.

Launch the backend software in separate terminals:
```
$ roslaunch pose_graph_backend pose_graph_node_euroc.launch
$ roslaunch voxblox_ros euroc_dataset.launch
```
Wait for the messages `[PGB] Sucessfully read the parameters and the vocabulary` and `[PGB] Set callbacks` to be printed in the terminal. This indicates that the pose-graph is ready.  

Navigate to the folder `pose_graph_backend/data` and play each bag file in separate terminals:
```
$ rosbag play MH_01_PreRecordedUAV.bag
$ rosbag play MH_02_PreRecordedUAV.bag
$ rosbag play MH_03_PreRecordedUAV.bag
```

### 2. General Usage ###
The following instructions show how to run the Pose Graph for a custom sensor set-up. First, you need to create the necessary configuration files as described above. Then, you need to adapt the launch files to the number of agents and their IDs and map the topics to the right names.  

To start up the Pose Graph, run the following launch files in separate terminals **on the server side**:
``` 
$ roslaunch pose_graph_backend pose_graph_node.launch num_agents:=NUM_AGENTS
$ roslaunch voxblox_ros voxblox_server_node.launch num_agents:=NUM_AGENTS
```
Wait for the message `[PGB] Sucessfully read the parameters and the vocabulary` to be printed in the terminal. This indicates that the pose-graph is ready. Note that the Voxblox node outputs visuals in RVIZ every 2 seconds.  

**On the robot's side**, run the following instructions in separate terminals for all the robots. The launch files **must be edited** to reflect assigned UAV IDs in the `agent_id` parameter:
```
$ roslaunch vins_estimator vins_estimator.launch
$ roslaunch pcl_fusion pcl_fusion_node.launch
```
The `pcl_fusion` launch file should launch `dense_stereo` to create a dense point cloud from stereo images. The UAVs should now be able to begin flying, and the agents' trajectories should appear in `RVIZ` after the GPS frame initialization has been completed in the Pose Graph.

## Contributing ##
Contributions that help to improve the code are welcome. In case you want to contribute, please adapt to the [Google C++ coding style](https://google.github.io/styleguide/cppguide.html) and run `bash clang-format-all .` on your code before any commit.

