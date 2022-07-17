#include "coxgraph/server/pose_graph_interface.h"

#include <vector>

#include <voxgraph/backend/constraint/relative_pose_constraint.h>

namespace coxgraph {
namespace server {

void PoseGraphInterface::addSubmap(SerSmId submap_id) {
  if (robocentric_) {
    voxgraph::PoseGraphInterface::addSubmap(submap_id);
  } else {
    // non-robocentric
    // Configure the submap node and add it to the pose graph
    voxgraph::SubmapNode::Config node_config = node_templates_.submap;
    node_config.submap_id = submap_id;
    CHECK(submap_collection_ptr_->getSubmapPose(submap_id,
                                                &node_config.T_I_node_initial));
    if (submap_id == 0) {
      ROS_INFO("Setting pose of submap 0 to constant");
      node_config.set_constant = true;
    } else {
      node_config.set_constant = false;
    }
    pose_graph_.addSubmapNode(node_config);
    ROS_INFO_STREAM_COND(verbose_,
                         "Added node to graph for submap: " << submap_id);
  }
}

void PoseGraphInterface::optimize(bool enable_registration) {
  pose_graph_.optimize(true);

  // update registration constrains after loop closure optimized. Submaps
  // overlapping can only be determined after their relative poses computed by
  // loop closure optimization
  if (enable_registration) updateRegistrationConstraints();

  // Optimize the pose graph with all constraints enabled
  pose_graph_.optimize();

  // Publish debug visuals
  if (pose_graph_pub_.getNumSubscribers() > 0) {
    LOG(INFO) << "publish pose graph: " << pose_graph_.getSubmapPoses().size();
    pose_graph_vis_.publishPoseGraph(pose_graph_, visualization_odom_frame_,
                                     "optimized", pose_graph_pub_);
  }
}

void PoseGraphInterface::updateSubmapRPConstraints() {
  resetSubmapRelativePoseConstrains();
  for (int cid = 0; cid < cox_submap_collection_ptr_->getClientNumber();
       cid++) {
    std::vector<SerSmId> cli_ser_sm_ids;
    if (!cox_submap_collection_ptr_->getSerSmIdsByCliId(cid, &cli_ser_sm_ids))
      continue;
    for (int i = 0; i < cli_ser_sm_ids.size() - 1; i++) {
      int j = i + 1;
      SerSmId sid_i = cli_ser_sm_ids.at(i);
      SerSmId sid_j = cli_ser_sm_ids.at(j);

      Transformation T_M_SMi =
          cox_submap_collection_ptr_->getSubmapPtr(sid_i)->getPose();
      Transformation T_M_SMj =
          cox_submap_collection_ptr_->getSubmapPtr(sid_j)->getPose();
      Transformation T_SMi_SMj = T_M_SMi.inverse() * T_M_SMj;
      addSubmapRelativePoseConstraint(sid_i, sid_j, T_SMi_SMj);
    }
  }
}

void PoseGraphInterface::addSubmapRelativePoseConstraint(
    const SerSmId& first_submap_id, const SerSmId& second_submap_id,
    const Transformation& T_S1_S2) {
  RelativePoseConstraint::Config submap_rp_config;
  submap_rp_config.information_matrix = sm_rp_info_matrix_;
  submap_rp_config.origin_submap_id = first_submap_id;
  submap_rp_config.destination_submap_id = second_submap_id;
  submap_rp_config.T_origin_destination = T_S1_S2;

  // Add the constraint to the pose graph
  // TODO(mikexyl): since these should be called every time submap pose updated,
  // don't log it
  pose_graph_.addSubmapRelativePoseConstraint(submap_rp_config);
}

void PoseGraphInterface::addForceRegistrationConstraint(
    const SerSmId& first_submap_id, const SerSmId& second_submap_id) {
  RegistrationConstraint::Config constraint_config =
      measurement_templates_.registration;
  constraint_config.first_submap_id = first_submap_id;
  constraint_config.second_submap_id = second_submap_id;

  // Add pointers to both submaps
  constraint_config.first_submap_ptr =
      submap_collection_ptr_->getSubmapConstPtr(first_submap_id);
  constraint_config.second_submap_ptr =
      submap_collection_ptr_->getSubmapConstPtr(second_submap_id);
  CHECK_NOTNULL(constraint_config.first_submap_ptr);
  CHECK_NOTNULL(constraint_config.second_submap_ptr);

  // Add the constraint to the pose graph
  pose_graph_.addForceRegistrationConstraint(constraint_config);
}

}  // namespace server
}  // namespace coxgraph
