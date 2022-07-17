#include <glog/logging.h>
#include <ros/ros.h>

#include "coxgraph/server/coxgraph_server.h"

int main(int argc, char** argv) {
  // Start logging
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();

  // Register with ROS master
  ros::init(argc, argv, "coxgraph_server");

  // Create node handles
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  // Create the mapper
  coxgraph::CoxgraphServer coxgraph_server(nh, nh_private);

  // Spin
  ros::spin();

  // Exit normally
  return 0;
}
