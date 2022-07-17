#ifndef COXGRAPH_SERVER_SUBMAP_COLLECTION_H_
#define COXGRAPH_SERVER_SUBMAP_COLLECTION_H_

#include <voxgraph/frontend/submap_collection/voxgraph_submap_collection.h>

#include <boost/filesystem/path.hpp>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "coxgraph/common.h"

namespace coxgraph {
namespace server {

class SubmapCollection : public voxgraph::VoxgraphSubmapCollection {
 public:
  typedef std::shared_ptr<SubmapCollection> Ptr;

  SubmapCollection(const voxgraph::VoxgraphSubmap::Config& submap_config,
                   int8_t client_number, bool verbose = false)
      : voxgraph::VoxgraphSubmapCollection(submap_config, verbose),
        client_number_(client_number) {}

  // Copy constructor without copy mutex
  SubmapCollection(const SubmapCollection& rhs)
      : voxgraph::VoxgraphSubmapCollection(
            static_cast<voxgraph::VoxgraphSubmapCollection>(rhs)),
        client_number_(rhs.client_number_),
        sm_cli_id_map_(rhs.sm_cli_id_map_),
        cli_ser_sm_id_map_(rhs.cli_ser_sm_id_map_),
        sm_id_ori_pose_map_(rhs.sm_id_ori_pose_map_) {}

  ~SubmapCollection() = default;

  const int8_t& getClientNumber() const { return client_number_; }

  Transformation addSubmap(const CliSm::Ptr& submap_ptr, const CliId& cid,
                           const CliSmId& cli_sm_id);

  inline bool getSerSmIdsByCliId(const CliId& cid,
                                 std::vector<SerSmId>* ser_sids) {
    if (cli_ser_sm_id_map_.count(cid)) {
      *ser_sids = cli_ser_sm_id_map_[cid];
      return true;
    } else {
      return false;
    }
  }

  inline bool getSerSmIdByCliSmId(const CliId& cid, const CliSmId& cli_sm_id,
                                  SerSmId* ser_sm_id) {
    CHECK(ser_sm_id != nullptr);
    for (auto ser_sm_id_v : cli_ser_sm_id_map_[cid]) {
      if (sm_cli_id_map_[ser_sm_id_v].second == cli_sm_id) {
        *ser_sm_id = ser_sm_id_v;
        return true;
      }
    }
    return false;
  }

  inline bool getCliSmIdsByCliId(const CliId& cid,
                                 std::vector<CliSmId>* cli_sids) {
    CHECK(cli_sids != nullptr);
    cli_sids->clear();
    for (auto ser_sm_id_v : cli_ser_sm_id_map_[cid]) {
      cli_sids->emplace_back(sm_cli_id_map_[ser_sm_id_v].second);
    }
    return !cli_sids->empty();
  }

  inline void updateOriPose(const SerSmId& ser_sm_id,
                            const Transformation& pose) {
    CHECK(exists(ser_sm_id));
    sm_id_ori_pose_map_[ser_sm_id] = pose;
  }
  inline Transformation getOriPose(const SerSmId& ser_sm_id) {
    CHECK(sm_id_ori_pose_map_.count(ser_sm_id));
    return sm_id_ori_pose_map_[ser_sm_id];
  }

  inline std::timed_mutex* getPosesUpdateMutex() {
    return &submap_poses_update_mutex;
  }

  CIdCSIdPair getCliIdPairBySsid(SerSmId ssid) {
    CHECK(sm_cli_id_map_.count(ssid));
    return sm_cli_id_map_[ssid];
  }

  VoxgraphSubmapCollection::PoseStampedVector getPoseHistory(CliId cid) {
    using PoseCountPair = std::pair<Transformation, int>;
    std::map<ros::Time, PoseCountPair> averaged_trajectory;
    // Iterate over all submaps and poses
    for (const auto& submap_ptr : getSubmapConstPtrs()) {
      if (getCliIdPairBySsid(submap_ptr->getID()).first != cid) continue;
      for (const std::pair<const ros::Time, Transformation>& time_pose_pair :
           submap_ptr->getPoseHistory()) {
        // Transform the pose from submap frame into odom frame
        const Transformation T_O_B_i =
            submap_ptr->getPose() * time_pose_pair.second;
        const ros::Time& timestamp_i = time_pose_pair.first;

        // Insert, or average if there was a previous pose with the same stamp
        auto it = averaged_trajectory.find(timestamp_i);
        if (it == averaged_trajectory.end()) {
          averaged_trajectory.emplace(timestamp_i, PoseCountPair(T_O_B_i, 1));
        } else {
          it->second.second++;
          const double lambda = 1.0 / it->second.second;
          it->second.first = kindr::minimal::interpolateComponentwise(
              it->second.first, T_O_B_i, lambda);
        }
      }
    }

    // Copy the averaged trajectory poses into the msg and compute the total
    // trajectory length
    PoseStampedVector poses;
    float total_trajectory_length = 0.0;
    Transformation::Position previous_position;
    bool previous_position_initialized = false;
    for (const auto& kv : averaged_trajectory) {
      geometry_msgs::PoseStamped pose_stamped_msg;
      pose_stamped_msg.header.stamp = kv.first;
      tf::poseKindrToMsg(kv.second.first.cast<double>(),
                         &pose_stamped_msg.pose);
      poses.emplace_back(pose_stamped_msg);

      if (previous_position_initialized) {
        total_trajectory_length +=
            (kv.second.first.getPosition() - previous_position).norm();
      } else {
        previous_position_initialized = true;
      }
      previous_position = kv.second.first.getPosition();
    }
    ROS_INFO_STREAM("Total trajectory length: " << total_trajectory_length);
    return poses;
  }

  void savePoseHistoryToFile(std::string file_path) {
    for (CliId cid = 0; cid < client_number_; cid++) {
      auto pose_history = getPoseHistory(cid);
      LOG(INFO) << cid << " " << pose_history.size();

      boost::filesystem::path p(file_path);
      p.append("opt_c" + std::to_string(cid) + ".txt");
      std::ofstream f;
      f.open(p.string());
      f << std::fixed;

      for (auto const& pose : pose_history) {
        f << std::setprecision(6) << pose.header.stamp.toSec()
          << std::setprecision(7) << " " << pose.pose.position.x << " "
          << pose.pose.position.y << " " << pose.pose.position.z << " "
          << pose.pose.orientation.x << " " << pose.pose.orientation.y << " "
          << pose.pose.orientation.z << " " << pose.pose.orientation.w
          << std::endl;
      }
      f.close();
    }
  }

 private:
  typedef std::pair<CliId, CliId> CliIdPair;
  typedef std::unordered_map<SerSmId, CIdCSIdPair> SmCliIdMap;
  typedef std::unordered_map<CliId, std::vector<SerSmId>> CliSerSmIdMap;

  Transformation mergeToCliMap(const CliSm::Ptr& submap_ptr);

  const int8_t client_number_;

  SmCliIdMap sm_cli_id_map_;
  CliSerSmIdMap cli_ser_sm_id_map_;

  std::unordered_map<SerSmId, Transformation> sm_id_ori_pose_map_;

  std::timed_mutex submap_poses_update_mutex;
};

}  // namespace server
}  // namespace coxgraph

#endif  // COXGRAPH_SERVER_SUBMAP_COLLECTION_H_
