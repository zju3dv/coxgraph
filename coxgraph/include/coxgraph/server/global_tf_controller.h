#ifndef COXGRAPH_SERVER_GLOBAL_TF_CONTROLLER_H_
#define COXGRAPH_SERVER_GLOBAL_TF_CONTROLLER_H_

#include <tf/transform_broadcaster.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "coxgraph/common.h"
#include "coxgraph/server/client_tf_optimizer.h"
#include "coxgraph/server/distribution/distribution_controller.h"
#include "coxgraph/server/pose_graph_interface.h"

namespace coxgraph {
namespace server {

class GlobalTfController {
 public:
  struct Config {
    Config() : init_cli_map_dist(10) {}
    int32_t init_cli_map_dist;

    friend inline std::ostream& operator<<(std::ostream& s, const Config& v) {
      s << std::endl
        << "Global Tf Controller using Config" << std::endl
        << "  Initial Client Mission Frame Distance: " << v.init_cli_map_dist
        << std::endl
        << "-------------------------------------------" << std::endl;
      return (s);
    }
  };

  static Config getConfigFromRosParam(const ros::NodeHandle& nh_private);

  typedef std::shared_ptr<GlobalTfController> Ptr;
  using PoseMap = ClientTfOptimizer::PoseMap;

  GlobalTfController(const ros::NodeHandle& nh,
                     const ros::NodeHandle& nh_private, int8_t client_number,
                     std::string map_fram_prefix,
                     DistributionController::Ptr distrib_ctl_ptr, bool verbose)
      : GlobalTfController(nh, nh_private, client_number, map_fram_prefix,
                           distrib_ctl_ptr, getConfigFromRosParam(nh_private),
                           verbose) {}

  GlobalTfController(const ros::NodeHandle& nh,
                     const ros::NodeHandle& nh_private, int8_t client_number,
                     std::string map_frame_prefix,
                     DistributionController::Ptr distrib_ctl_ptr,
                     const Config& config, bool verbose)
      : verbose_(verbose),
        nh_(nh),
        nh_private_(nh_private),
        client_number_(client_number),
        map_frame_prefix_(map_frame_prefix),
        distrib_ctl_ptr_(distrib_ctl_ptr),
        config_(config),
        global_mission_frame_(map_frame_prefix + "_g"),
        client_tf_optimizer_(nh_private, verbose),
        pose_updated_(false) {
    LOG(INFO) << config;
    initCliMapPose();
  }

  ~GlobalTfController() = default;

  const std::string& getGlobalMissionFrame() const {
    return global_mission_frame_;
  }

  void publishTfGloCli();

  void addCliMapRelativePose(const CliId& first_cid, const CliId& second_cid,
                             const Transformation& T_C1_C2);

  void resetCliMapRelativePoses() {
    client_tf_optimizer_.resetClientRelativePoseConstraints();
  }

  std::mutex* getPoseUpdateMutex() { return &pose_update_mutex; }

  bool inControl() const { return distrib_ctl_ptr_->inControl(); }

  const tf::StampedTransform& getTGCliOpt(const CliId& cid) const {
    return T_G_CLI_opt_[cid];
  }

  bool ifClientFused(CliId cid) const { return cli_tf_fused_[cid]; }

 private:
  void initCliMapPose();
  void pubCliTfCallback(const ros::TimerEvent& event);
  void updateCliMapPose();
  void computeOptCliMapPose();

  bool verbose_;

  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  Config config_;
  std::string map_frame_prefix_;

  const int8_t client_number_;
  const std::string global_mission_frame_;
  std::vector<std::string> cli_mission_frames_;

  ros::Timer tf_pub_timer_;
  tf::TransformBroadcaster tf_boardcaster_;
  std::vector<bool> cli_tf_fused_;
  std::vector<tf::StampedTransform> T_G_CLI_opt_;

  ClientTfOptimizer client_tf_optimizer_;
  bool pose_updated_;
  std::mutex pose_update_mutex;

  DistributionController::Ptr distrib_ctl_ptr_;

  constexpr static float kTfPubFreq = 100;
};

}  // namespace server
}  // namespace coxgraph

#endif  // COXGRAPH_SERVER_GLOBAL_TF_CONTROLLER_H_
