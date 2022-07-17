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

1. Ubuntu 18;
2. ROS Melodic;
3. OpenCV 3;
4. Open3D 0.10.

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

3. Clone the repository and dependencies:

        cd src
        git clone git@github.com:zju3dv/coxgraph.git
        wstool init
        wstool merge coxgraph/coxgraph_ssh.rosinstall
        wstool update

4. Start building:

        catkin build coxgraph vins_client_server pose_graph_backend image_undistort

### SLAM Frontend Selection

A convenient interface enabling any multi-robot SLAM system to cooperate with **Coxgraph** is provided named as `coxgraph_mod`. One can easily insert a few lines into other frontend to use it with **Coxgraph**. In this code, we used **vins_client_server** as the frontend.

### Experiment Machine Hall

1. Download EuRoC bag files [here](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets).
2. In this code, we run 2-agent collaborative session on the same machine. If you run 3-agent simultaneously, the performence of **Coxgraph** might degrade since the computational resources are overloaded. Run the launch file:

        roslaunch coxgraph run_experiment_euroc.launch
        roslaunch coxgraph coxgraph_rviz.launch
