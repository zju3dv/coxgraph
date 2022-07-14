#ifndef VOXGRAPH_FRONTEND_SUBMAP_FITNESS_EVALUATION_H_
#define VOXGRAPH_FRONTEND_SUBMAP_FITNESS_EVALUATION_H_

#include "voxgraph/common.h"
#include "voxgraph/frontend/submap_collection/voxgraph_submap.h"

#include <algorithm>
#include <utility>

namespace voxgraph {
class SubmapFitnessEvalution {
 public:
  struct Config {
    float max_valid_distance = 0.03;
    bool only_isopoints = true;
    float k_overlap = 0.30;
    float max_traj_distance = 0.05;
    float k_traj = 0.60;

    friend inline std::ostream& operator<<(std::ostream& s, const Config& v) {
      s << std::endl
        << "SubmapFitnessEvalution using Config:" << std::endl
        << "  max_valid_distance: " << v.max_valid_distance << std::endl
        << "  only_isopoints: " << v.only_isopoints << std::endl
        << "  k_overlap: " << v.k_overlap << std::endl
        << "  max_traj_distance: " << v.max_traj_distance << std::endl
        << "  k_traj: " << v.k_traj << std::endl
        << "-------------------------------------------" << std::endl;
      return (s);
    }
  };
  static Config getConfigFromRosParam(const ros::NodeHandle& nh_private) {
    Config config;
    nh_private.param("max_valid_distance", config.max_valid_distance,
                     config.max_valid_distance);
    nh_private.param("only_isopoints", config.only_isopoints,
                     config.only_isopoints);
    nh_private.param("k_overlap", config.k_overlap, config.k_overlap);
    nh_private.param("k_traj", config.k_traj, config.k_traj);
    return config;
  }

  explicit SubmapFitnessEvalution(const ros::NodeHandle& nh_private)
      : config_(getConfigFromRosParam(nh_private)) {}
  virtual ~SubmapFitnessEvalution() = default;

  std::pair<bool, float> evaluateFitness(const VoxgraphSubmap& submap_a,
                                         const VoxgraphSubmap& submap_b,
                                         Transformation T_A_B) {
    // return std::make_pair(true, 0);
    LOG(INFO) << "checking fitness "
              << submap_a.getTsdfMap().getTsdfLayer().getMemorySize() << " "
              << submap_b.getTsdfMap().getTsdfLayer().getMemorySize();
    auto result_points = evaluateFitnessPoints(submap_a, submap_b, T_A_B);
    auto result_traj = evaluateFitnessPoseHistory(submap_a, submap_b, T_A_B);
    return std::make_pair(result_points.first && result_traj.first, 0);
  }

  std::pair<bool, float> evaluateFitnessPoseHistory(
      const VoxgraphSubmap& submap_a, const VoxgraphSubmap& submap_b,
      Transformation T_A_B) {
    LOG(INFO) << T_A_B;
    auto pose_history_a = submap_a.getPoseHistory();
    auto pose_history_b = submap_b.getPoseHistory();
    LOG(INFO) << "traj----------------------------";

    int good = 0;
    for (auto const& pt : pose_history_a) {
      // std::cout << pt.second.getPosition() << std::endl;
      voxblox::Pointcloud T_B_APt;
      voxblox::transformPointcloud(T_A_B.inverse(), {pt.second.getPosition()},
                                   &T_B_APt);
      if (!checkDistanceValid(submap_b.getTsdfMap().getTsdfLayer(), T_B_APt[0],
                              0, config_.max_traj_distance)) {
        good++;
        //  LOG(INFO) << submap_b.getTsdfMap()
        //                   .getTsdfLayer()
        //                   .getVoxelPtrByCoordinates(T_B_APt[0])
        //                   ->distance;
        //  LOG(WARNING) << "traj eval failed";
        // return std::make_pair(false, 0);
      }
    }
    LOG(INFO) << static_cast<float>(good) / pose_history_a.size();
    if (static_cast<float>(good) / pose_history_a.size() < config_.k_traj) {
      LOG(WARNING) << "traj eval failed";
      return std::make_pair(false,
                            static_cast<float>(good) / pose_history_a.size());
    }

    LOG(INFO) << "***********";
    good = 0;
    for (auto const& pt : pose_history_b) {
      //   std::cout << pt.second.getPosition() << std::endl;
      voxblox::Pointcloud T_B_APt;
      voxblox::transformPointcloud(T_A_B, {pt.second.getPosition()}, &T_B_APt);
      if (!checkDistanceValid(submap_a.getTsdfMap().getTsdfLayer(), T_B_APt[0],
                              0, config_.max_traj_distance)) {
        good++;
        //   LOG(WARNING) << "traj eval failed";
        // return std::make_pair(false, 0);
      }
    }

    LOG(INFO) << static_cast<float>(good) / pose_history_b.size();
    if (static_cast<float>(good) / pose_history_b.size() < config_.k_traj) {
      LOG(WARNING) << "traj eval failed";
      return std::make_pair(false,
                            static_cast<float>(good) / pose_history_b.size());
    }

    LOG(WARNING) << "traj eval succeed";
    return std::make_pair(true, 0);
  }

  std::pair<bool, float> evaluateFitnessPoints(const VoxgraphSubmap& submap_a,
                                               const VoxgraphSubmap& submap_b,
                                               Transformation T_A_B) {
    auto const& registration_points_a = submap_a.getRegistrationPoints(
        VoxgraphSubmap::RegistrationPointType::kIsosurfacePoints);
    auto const& registration_points_b = submap_b.getRegistrationPoints(
        VoxgraphSubmap::RegistrationPointType::kIsosurfacePoints);

    int num_valid = 0;
    int total_points = 0;
    for (auto const& pt : registration_points_a.getItem()) {
      total_points++;
      voxblox::Pointcloud T_B_APt;
      voxblox::transformPointcloud(T_A_B.inverse(), {pt.position}, &T_B_APt);
      if (checkDistanceValid(submap_b.getTsdfMap().getTsdfLayer(), T_B_APt[0],
                             pt.distance, config_.max_valid_distance))
        num_valid++;
    }
    if (total_points == 0) {
      LOG(WARNING) << "points eval failed: no valid points";
      return std::make_pair(false, 0.0);
    }
    float fitness_0 = static_cast<float>(num_valid) / total_points;
    if (fitness_0 < config_.k_overlap) {
      LOG(WARNING) << "points eval failed: " << fitness_0;
      return std::make_pair(false, fitness_0);
    }

    num_valid = 0;
    total_points = 0;
    for (auto const& pt : registration_points_b.getItem()) {
      total_points++;
      voxblox::Pointcloud T_B_APt;
      voxblox::transformPointcloud(T_A_B, {pt.position}, &T_B_APt);
      if (checkDistanceValid(submap_a.getTsdfMap().getTsdfLayer(), T_B_APt[0],
                             pt.distance, config_.max_valid_distance))
        num_valid++;
    }
    if (total_points == 0) return std::make_pair(false, 0.0);
    float fitness_1 = static_cast<float>(num_valid) / total_points;

    if (fitness_1 > config_.k_overlap) {
      LOG(INFO) << "points eval succeeded: " << fitness_1;
      return std::make_pair(true, std::min(fitness_0, fitness_1));

    } else {
      LOG(WARNING) << "points eval failed: " << fitness_1;
      return std::make_pair(false, fitness_1);
    }
  }

  bool checkDistanceValid(const voxblox::Layer<voxblox::TsdfVoxel>& tsdf_layer,
                          voxblox::Point point, float distance,
                          float max_valid_distance) {
    auto tsdf_voxel = tsdf_layer.getVoxelPtrByCoordinates(point);
    if (!tsdf_voxel) return false;
    if (std::abs(tsdf_voxel->distance - distance) < max_valid_distance)
      return true;
    return false;
  }

 private:
  Config config_;
};
}  // namespace voxgraph

#endif  // VOXGRAPH_FRONTEND_SUBMAP_FITNESS_EVALUATION_H_
