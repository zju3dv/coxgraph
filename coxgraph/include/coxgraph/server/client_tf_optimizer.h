#ifndef COXGRAPH_SERVER_CLIENT_TF_OPTIMIZER_H_
#define COXGRAPH_SERVER_CLIENT_TF_OPTIMIZER_H_

#include <ros/ros.h>

#include <map>

#include "coxgraph/common.h"
#include "coxgraph/server/backend/client_frame_node.h"
#include "coxgraph/server/backend/pose_graph.h"
#include "coxgraph/utils/ros_params.h"

namespace coxgraph {
namespace server {

class ClientTfOptimizer {
 public:
  using PoseMap = PoseGraph::PoseMap;

  ClientTfOptimizer(const ros::NodeHandle& nh_private, bool verbose)
      : verbose_(verbose) {
    utils::setInformationMatrixFromRosParams(
        ros::NodeHandle(nh_private,
                        "client_map_relative_pose/information_matrix"),
        &cli_rp_info_matrix_);
  }

  void addClient(const CliId& cid, const Transformation& pose);

  void addClientRelativePoseMeasurement(const CliId& first_cid,
                                        const CliId& second_cid,
                                        const Transformation& T_C1_C2);

  void resetClientRelativePoseConstraints() {
    pose_graph_.resetClientRelativePoseConstraint();
  }

  void optimize() { pose_graph_.optimize(); }

  PoseMap getClientMapTfs() { return pose_graph_.getClientMapTf(); }

 private:
  bool verbose_;

  PoseGraph pose_graph_;

  InformationMatrix cli_rp_info_matrix_;
};

}  // namespace server
}  // namespace coxgraph
#endif  // COXGRAPH_SERVER_CLIENT_TF_OPTIMIZER_H_
