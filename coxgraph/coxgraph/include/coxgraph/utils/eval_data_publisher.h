#ifndef COXGRAPH_UTILS_EVAL_DATA_PUBLISHER_H_
#define COXGRAPH_UTILS_EVAL_DATA_PUBLISHER_H_

#include <node_evaluator/Bandwidth.h>
#include <ros/ros.h>

#include <string>
#include <utility>

namespace coxgraph {
namespace utils {

class EvalDataPublisher {
 public:
  EvalDataPublisher(const ros::NodeHandle& nh,
                    const ros::NodeHandle& nh_private)
      : nh_(nh), nh_private_(nh_private) {
    bw_pub_ = nh_private_.advertise<node_evaluator::Bandwidth>(
        "service_bandwidth", 10, true);
  }
  ~EvalDataPublisher() = default;

  void publishBandwidth(std::string name, uint64_t size, ros::Time time0,
                        ros::Time time1) {
    node_evaluator::Bandwidth bw_msg;
    bw_msg.name = name;
    bw_msg.size = size;
    bw_msg.time[0] = time0;
    bw_msg.time[1] = time1;
    bw_pub_.publish(bw_msg);
  }

 private:
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  ros::Publisher bw_pub_;
};
}  // namespace utils
}  // namespace coxgraph

#endif  // COXGRAPH_UTILS_EVAL_DATA_PUBLISHER_H_
