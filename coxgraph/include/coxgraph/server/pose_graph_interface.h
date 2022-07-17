#ifndef COXGRAPH_SERVER_POSE_GRAPH_INTERFACE_H_
#define COXGRAPH_SERVER_POSE_GRAPH_INTERFACE_H_

#include <visualization_msgs/Marker.h>
#include <voxgraph/backend/constraint/relative_pose_constraint.h>
#include <voxgraph/frontend/pose_graph_interface/pose_graph_interface.h>
#include <voxgraph/frontend/submap_collection/voxgraph_submap_collection.h>

#include <memory>
#include <string>

#include "coxgraph/common.h"
#include "coxgraph/server/submap_collection.h"
#include "coxgraph/utils/ros_params.h"

namespace coxgraph {
namespace server {

class PoseGraphInterface : public voxgraph::PoseGraphInterface {
 public:
  typedef std::shared_ptr<PoseGraphInterface> Ptr;

  using VoxgraphSubmapCollection = voxgraph::VoxgraphSubmapCollection;
  using RelativePoseConstraint = voxgraph::RelativePoseConstraint;
  using RegistrationConstraint = voxgraph::RegistrationConstraint;
  using PoseMap = voxgraph::PoseGraph::PoseMap;

  PoseGraphInterface(ros::NodeHandle nh_private,
                     const SubmapCollection::Ptr& submap_collection_ptr,
                     const MeshIntegratorConfig& mesh_config,
                     const std::string& visualizations_mission_frame,
                     bool robocentric, bool verbose = false,
                     std::string pose_graph_topic = "")
      : voxgraph::PoseGraphInterface(
            nh_private,
            static_cast<VoxgraphSubmapCollection::Ptr>(submap_collection_ptr),
            mesh_config, visualizations_mission_frame, verbose),
        cox_submap_collection_ptr_(submap_collection_ptr),
        robocentric_(robocentric) {
    utils::setInformationMatrixFromRosParams(
        ros::NodeHandle(nh_private, "submap_relative_pose/information_matrix"),
        &sm_rp_info_matrix_);

    if (pose_graph_topic.size()) {
      pose_graph_pub_.shutdown();
      pose_graph_pub_ = nh_private.advertise<visualization_msgs::Marker>(
          "pose_graph_global", 100, true);
    }
  }

  // Copy constructor
  PoseGraphInterface(const PoseGraphInterface& rhs) = default;

  // Copy constructor with a new submap collection ptr
  PoseGraphInterface(const PoseGraphInterface& rhs,
                     SubmapCollection::Ptr submap_collection_ptr)
      : PoseGraphInterface(rhs) {
    // reset submap collection ptr
    submap_collection_ptr_ =
        static_cast<VoxgraphSubmapCollection::Ptr>(submap_collection_ptr);
    cox_submap_collection_ptr_ = submap_collection_ptr;
  }

  ~PoseGraphInterface() = default;

  void addSubmap(SerSmId submap_id);

  void optimize(bool enable_registration);

  void updateSubmapRPConstraints();

  void resetSubmapRelativePoseConstrains() {
    pose_graph_.resetSubmapRelativePoseConstraints();
  }

  void addSubmapRelativePoseConstraint(const SerSmId& first_submap_id,
                                       const SerSmId& second_submap_id,
                                       const Transformation& T_S1_S2);

  void addForceRegistrationConstraint(const SerSmId& first_submap_id,
                                      const SerSmId& second_submap_id);

  PoseMap getPoseMap() { return pose_graph_.getSubmapPoses(); }

  void printResiduals(ConstraintType constraint_type) {
    for (double residual : evaluateResiduals(constraint_type)) {
      std::cout << residual << " ";
    }
    std::cout << std::endl;
  }

 private:
  bool robocentric_;

  SubmapCollection::Ptr cox_submap_collection_ptr_;
  InformationMatrix sm_rp_info_matrix_;
};

}  // namespace server
}  // namespace coxgraph

#endif  // COXGRAPH_SERVER_POSE_GRAPH_INTERFACE_H_
