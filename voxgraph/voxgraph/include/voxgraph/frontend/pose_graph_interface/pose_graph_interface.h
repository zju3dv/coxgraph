#ifndef VOXGRAPH_FRONTEND_POSE_GRAPH_INTERFACE_POSE_GRAPH_INTERFACE_H_
#define VOXGRAPH_FRONTEND_POSE_GRAPH_INTERFACE_POSE_GRAPH_INTERFACE_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "voxgraph/backend/pose_graph.h"
#include "voxgraph/common.h"
#include "voxgraph/frontend/pose_graph_interface/measurement_templates.h"
#include "voxgraph/frontend/pose_graph_interface/node_templates.h"
#include "voxgraph/frontend/submap_collection/voxgraph_submap_collection.h"
#include "voxgraph/frontend/submap_fitness_evaluation.h"
#include "voxgraph/tools/visualization/pose_graph_visuals.h"
#include "voxgraph/tools/visualization/submap_visuals.h"

namespace voxgraph {
class PoseGraphInterface {
 public:
  typedef std::shared_ptr<PoseGraphInterface> Ptr;
  typedef std::vector<SubmapIdPair> OverlappingSubmapList;

  explicit PoseGraphInterface(
      ros::NodeHandle node_handle,
      VoxgraphSubmapCollection::Ptr submap_collection_ptr,
      voxblox::MeshIntegratorConfig mesh_config,
      const std::string& visualizations_odom_frame, bool verbose = false);

  void setVerbosity(bool verbose) { verbose_ = verbose; }
  void setMeasurementConfigFromRosParams(const ros::NodeHandle& node_handle) {
    measurement_templates_.setFromRosParams(node_handle);
  }

  virtual void addSubmap(SubmapID submap_id);

  // Method to recalculate which submaps overlap and update their
  // registration constraints accordingly
  void updateRegistrationConstraints();

  // NOTE: The pose graph optimization works in 4D. Therefore the
  //       pitch and roll components of T_S1_S2 are simply ignored
  //       by the RelativePoseCostFunction.
  void addOdometryMeasurement(const SubmapID& first_submap_id,
                              const SubmapID& second_submap_id,
                              const Transformation& T_S1_S2);
  bool addLoopClosureMeasurement(const SubmapID& from_submap,
                                 const SubmapID& to_submap,
                                 const Transformation& transform,
                                 bool consistency_check = false);
  bool checkLoopClosureCandidates() {
    bool add_new_loop = false;
    for (auto& kv : loop_candidates_) {
      bool added = false;
      if (kv.second.size() < 2) continue;
      for (size_t i = 0; i < kv.second.size() - 1; i++) {
        bool consistent = true;
        for (size_t j = i + 1; i < kv.second.size(); j++) {
          auto res = kv.second[i].inverse() * kv.second[j];
          if (res.getPosition().norm() > 0.04 ||
              std::abs(res.getRotation().w()) > 0.005) {
            consistent = false;
            break;
          }
        }
        if (!consistent) {
          kv.second.erase(kv.second.begin() + i);
          i--;
        } else {
          addLoopClosureMeasurement(kv.first.first, kv.first.second,
                                    kv.second[i]);
          add_new_loop = true;
          added = true;
        }
      }
      if (added) loop_candidates_.erase(kv.first);
    }
    return add_new_loop;
  }
  void addGpsMeasurement() {}
  void addHeightMeasurement(const SubmapID& submap_id, const double& height);

  void optimize(float parameter_tolerance = 3e-3);

  void updateSubmapCollectionPoses();

  const OverlappingSubmapList& getOverlappingSubmapList() const {
    return overlapping_submap_list_;
  }

  bool getEdgeCovarianceMap(
      PoseGraph::EdgeCovarianceMap* edge_covariance_map_ptr) const;

  const PoseGraph::SolverSummaryList& getSolverSummaries() const {
    return pose_graph_.getSolverSummaries();
  }

  using ConstraintType = PoseGraph::ConstraintType;
  std::vector<double> evaluateResiduals(ConstraintType constraint_type) {
    return pose_graph_.evaluateResiduals(constraint_type);
  }

  void setSubmapCollectionPtr(
      const VoxgraphSubmapCollection::Ptr& submap_collection_ptr) {
    submap_collection_ptr_ = submap_collection_ptr;
  }

  bool hasSubmapNode(const voxgraph::SubmapNode::SubmapId& submap_id) {
    return pose_graph_.hasSubmapNode(submap_id);
  }

 protected:
  bool verbose_;

  VoxgraphSubmapCollection::Ptr submap_collection_ptr_;

  // Pose graph and visualization tools
  const std::string visualization_odom_frame_;
  PoseGraph pose_graph_;
  SubmapVisuals submap_vis_;
  PoseGraphVisuals pose_graph_vis_;
  ros::Publisher pose_graph_pub_;
  ros::Publisher submap_pub_;

  // Node and measurement config templates
  NodeTemplates node_templates_;
  MeasurementTemplates measurement_templates_;

  // Keep track of which submaps overlap
  OverlappingSubmapList overlapping_submap_list_;
  void updateOverlappingSubmapList();

  // K
  // eep track of whether new loop closures have been added
  // since the pose graph was last optimized
  bool new_loop_closures_added_since_last_optimization_;

  // Helper to add reference frames to the pose graph
  void addReferenceFrameIfMissing(ReferenceFrameNode::FrameId frame_id);

  SubmapFitnessEvalution fitness_eval_;
  std::map<SubmapIdPair, std::vector<Transformation>> loop_candidates_;
};
}  // namespace voxgraph

#endif  // VOXGRAPH_FRONTEND_POSE_GRAPH_INTERFACE_POSE_GRAPH_INTERFACE_H_
