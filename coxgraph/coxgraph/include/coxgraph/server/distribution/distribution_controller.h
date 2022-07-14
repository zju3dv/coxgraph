#ifndef COXGRAPH_SERVER_DISTRIBUTION_DISTRIBUTION_CONTROLLER_H_
#define COXGRAPH_SERVER_DISTRIBUTION_DISTRIBUTION_CONTROLLER_H_

#include <coxgraph_msgs/ControlTrigger.h>
#include <coxgraph_msgs/StateQuery.h>

#include <memory>
#include <string>

#include "coxgraph/common.h"
#include "coxgraph/server/submap_collection.h"
#include "coxgraph/utils/msg_converter.h"

namespace coxgraph {
namespace server {

class DistributionController {
 public:
  struct Config {
    Config() {}
    friend inline std::ostream& operator<<(std::ostream& s, const Config& v) {
      s << std::endl
        << "Distribution Controller using Config:" << std::endl
        << std::endl
        << "-------------------------------------------" << std::endl;
      return (s);
    }
  };

  static Config getConfigFromRosParam(const ros::NodeHandle& nh_private);

  typedef std::shared_ptr<DistributionController> Ptr;

  DistributionController(const ros::NodeHandle& nh,
                         const ros::NodeHandle& nh_private,
                         const SubmapCollection::Ptr& submap_collection_ptr)
      : nh_(nh),
        nh_private_(nh_private),
        submap_collection_ptr_(submap_collection_ptr) {
    nh_private_.param<bool>("in_control", in_control_, true);
    LOG(INFO) << "Server in control: "
              << static_cast<std::string>(in_control_ ? "true" : "false");

    advertiseTopics();
  }

  ~DistributionController() = default;

  void advertiseTopics() {
    control_trigger_srv_ = nh_private_.advertiseService(
        "control_trigger", &DistributionController::ControlTriggerCallback,
        this);
    state_query_srv_ = nh_private_.advertiseService(
        "state_query", &DistributionController::StateQueryCallback, this);
  }

  inline bool inControl() const { return in_control_; }

 private:
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  // Control trigger service
  ros::ServiceServer control_trigger_srv_;
  bool in_control_;
  bool ControlTriggerCallback(
      coxgraph_msgs::ControlTrigger::Request& request,      // NOLINT
      coxgraph_msgs::ControlTrigger::Response& response) {  // NOLINT
    LOG(INFO) << "Triggering control state to: "
              << static_cast<std::string>(request.in_control ? "true"
                                                             : "false");
    in_control_ = request.in_control;
    return true;
  }

  ros::ServiceServer state_query_srv_;
  SubmapCollection::Ptr submap_collection_ptr_;
  bool StateQueryCallback(
      coxgraph_msgs::StateQuery::Request& request,      // NOLINT
      coxgraph_msgs::StateQuery::Response& response) {  // NOLINT
    response.n_submaps = submap_collection_ptr_->getSubmapPtrs().size();
    for (auto const& submap_ptr : submap_collection_ptr_->getSubmapPtrs()) {
      response.bb.emplace_back(
          utils::msgFromBb(submap_ptr->getOdomFrameSurfaceAabb()));
    }
    return true;
  }
};

}  // namespace server
}  // namespace coxgraph

#endif  // COXGRAPH_SERVER_DISTRIBUTION_DISTRIBUTION_CONTROLLER_H_
