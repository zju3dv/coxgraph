#ifndef COXGRAPH_MOD_VIO_INTERFACE_H_
#define COXGRAPH_MOD_VIO_INTERFACE_H_

#include <std_srvs/SetBool.h>
#include <Eigen/Dense>

#include <map>
#include <string>

#include "coxgraph_mod/common.h"
#include "coxgraph_mod/loop_closure_publisher.h"
#include "coxgraph_mod/tf_publisher.h"

namespace coxgraph {
namespace mod {

class VIOInterface {
 public:
  VIOInterface(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
      : nh_(nh), nh_private_(nh_private) {
    loop_closure_pub_.reset(new LoopClosurePublisher(nh_, nh_private_));
    std::string toggle_mapping_srv_name;
    nh_private_.param<std::string>("toggle_mapping_srv_name",
                                   toggle_mapping_srv_name,
                                   toggle_mapping_srv_name);
    if (toggle_mapping_srv_name.size()) {
      toggle_mapping_srv_ =
          nh_.serviceClient<std_srvs::SetBool>(toggle_mapping_srv_name);
    }
  }

  ~VIOInterface() = default;

  void updatePose(Eigen::Matrix4d pose, double timestamp) {
    init(InitModule::tf);
    if (tf_pub_ == nullptr) return;
    tf_pub_->updatePose(pose, timestamp);
  }

  void updatePose(cv::Mat pose, double timestamp) {
    init(InitModule::tf);
    if (tf_pub_ == nullptr) return;
    tf_pub_->updatePose(pose, timestamp);
  }

  void publishLoopClosure(size_t from_client_id, double from_timestamp,
                          size_t to_client_id, double to_timestamp,
                          Eigen::Matrix4d T_A_B) {
    init(InitModule::lc);
    loop_closure_pub_->publishLoopClosure(from_client_id, from_timestamp,
                                          to_client_id, to_timestamp, T_A_B);
  }

  void publishLoopClosure(size_t from_client_id, double from_timestamp,
                          size_t to_client_id, double to_timestamp, cv::Mat R,
                          cv::Mat t) {
    init(InitModule::lc);
    loop_closure_pub_->publishLoopClosure(from_client_id, from_timestamp,
                                          to_client_id, to_timestamp, R, t);
  }

  void publishLoopClosure(CliId cid, const double& from_timestamp,
                          const double& to_timestamp, Eigen::Matrix4d T_A_B) {
    init(InitModule::lc);
    loop_closure_pub_->publishLoopClosure(cid, from_timestamp, cid,
                                          to_timestamp, T_A_B);
  }

  void publishLoopClosure(CliId cid, const double& from_timestamp,
                          const double& to_timestamp, cv::Mat R, cv::Mat t) {
    init(InitModule::lc);
    loop_closure_pub_->publishLoopClosure(cid, from_timestamp, to_timestamp, R,
                                          t);
  }

  void publishLoopClosure(const double& from_timestamp,
                          const double& to_timestamp, cv::Mat R, cv::Mat t) {
    init(InitModule::lc);
    loop_closure_pub_->publishLoopClosure(-1, from_timestamp, to_timestamp, R,
                                          t);
  }

  void publishLoopClosure(const double& from_timestamp,
                          const double& to_timestamp, Eigen::Matrix4d T_A_B) {
    init(InitModule::lc);
    loop_closure_pub_->publishLoopClosure(-1, from_timestamp, to_timestamp,
                                          T_A_B);
  }

  bool toggleMapping(bool b_mapping) {
    init(InitModule::mapping);
    std_srvs::SetBool toggle_mapping_msg;
    toggle_mapping_msg.request.data = b_mapping;
    if (toggle_mapping_srv_.call(toggle_mapping_msg)) {
      return true;
    } else {
      ROS_ERROR_STREAM(
          "Toggle Mapping to "
          << static_cast<std::string>(b_mapping ? "enabled" : "disabled")
          << " Failed");
      return false;
    }
  }

  bool needToFuse(CliId cid_a, CliId cid_b) {
    init(InitModule::lc);
    return loop_closure_pub_->needToFuseCached(cid_a, cid_b);
  }

  void updateNeedToFuse() {
    init(InitModule::lc);
    loop_closure_pub_->updateNeedToFuse();
  }

 private:
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  enum InitModule { nh = 0, tf, lc, mapping };

  std::map<InitModule, bool> initialized_;

  void init(InitModule init_module) {
    if (initialized_[init_module]) return;
    switch (init_module) {
      case InitModule::tf:

        tf_pub_.reset(new TfPublisher(nh_, nh_private_));
        initialized_[init_module] = true;
        break;
      default:
        break;
    }
  }

  TfPublisher::Ptr tf_pub_;
  LoopClosurePublisher::Ptr loop_closure_pub_;

  ros::ServiceClient toggle_mapping_srv_;
};

}  // namespace mod
}  // namespace coxgraph

#endif  // COXGRAPH_MOD_VIO_INTERFACE_H_
