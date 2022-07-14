# Coxgraph

**Coxgraph** is a multi robot collaborative reconstruction system providing realtime, low bandwidth and global consistent dense mapping.
As the name indicates, **Coxgraph** is heavily inspired by and adapted from [**Voxgraph**](https://github.com/ethz-asl/voxgraph.git).
We extend it to multi robot scenarios, and reduce the bandwidth of submap transmission by convert TSDF submaps to *mesh packs* and recover it back in the server.
Then check validity of the loop closure matches and optimization based on dense submaps.

[![Video player thumbnail for GitHub](https://raw.githubusercontent.com/LXYYY/lxyyy.github.io/master/images/coxgraph_video_thumbnail.jpg)](https://youtu.be/KgPLRP_ADQQ)

## Citing

This work is in IROS 2021 Best Paper Award Finalist on Safety, Security, and Rescue Robotics.
You can download the pdf from [here](http://www.cad.zju.edu.cn/home/gfzhang/papers/Coxgraph/IROS21_Coxgraph.pdf).
 When using **Coxgraph** in your research, please cite our publication:


```
@inproceedings{xiangyu2021coxgraph,
  author={Liu, Xiangyu and Ye, Weicai and Tian, Chaoran and Cui, Zhaopeng and Zhang, Guofeng and Bao, Hujun},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  title={Coxgraph: Multi-Robot Collaborative, Globally Consistent, Online Dense Reconstruction System},
  year={2021}
}
```

## Installation

### Requirements

1. Ubuntu 18 or 20;
2. ROS Melodic or Noetic
3. OpenCV 3 or 4;
4. Open3D 0.11 or newer.

### Build

1. Install workspace and catkin tools:

        sudo apt install python-catkin-tools python-wstool

2. Create and config a catkin workspace:

        mkdir -p ~/catkin_ws/src
        cd ~/catkin_ws
        catkin init
        catkin config --extend /opt/ros/${ROS_DISTRO}
        catkin config --merge-devel
        catkin config -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=14

3. Start building:

        catkin build coxgraph vins_client_server pose_graph_backend image_undistort

### Run
        roslaunch coxgraph run_experiment_euroc.launch 
        roslaunch coxgraph coxgraph_rviz.launch
	rosservice call /coxgraph/coxgraph_server_node/get_final_global_mesh "filepath=<result_dir>"
        (you need to wait to save the mesh)

### SLAM Frontend Selection

A convenient interface enabling any multi-robot SLAM system to cooperate with **Coxgraph** is provided named as `coxgraph_mod`. One can easily insert a few lines into other frontend to use it with **Coxgraph**.
So far, there are three frontends we can use:

1. **vins_client_server**, used in the demo;
2. **CORB_SLAM**
3. We also implement a Multi-Robot version of **rovioli** . (in production)

### Experiment Machine Hall

1. Download EuRoC bag files [here](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets).
2. Run the launch file:

        roslaunch coxgraph run_experiment_euroc.launch
