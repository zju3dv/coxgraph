#ifndef COXGRAPH_SERVER_COXGRAPH_SERVER_H_
#define COXGRAPH_SERVER_COXGRAPH_SERVER_H_

#include <coxgraph_msgs/ControlTrigger.h>
#include <coxgraph_msgs/FilePath.h>
#include <coxgraph_msgs/MapFusion.h>
#include <coxgraph_msgs/NeedToFuseSrv.h>
#include <ros/ros.h>
#include <voxblox_ros/ros_params.h>
#include <voxgraph/frontend/pose_graph_interface/pose_graph_interface.h>
#include <voxgraph/frontend/submap_collection/voxgraph_submap.h>
#include <voxgraph/frontend/submap_collection/voxgraph_submap_collection.h>
#include <voxgraph/tools/ros_params.h>
#include <voxgraph/tools/threading_helper.h>
#include <voxgraph/tools/visualization/submap_visuals.h>
#include <voxgraph_msgs/LoopClosure.h>

#include <deque>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "coxgraph/common.h"
#include "coxgraph/server/client_handler.h"
#include "coxgraph/server/distribution/distribution_controller.h"
#include "coxgraph/server/global_tf_controller.h"
#include "coxgraph/server/pose_graph_interface.h"
#include "coxgraph/server/submap_collection.h"
#include "coxgraph/server/visualizer/server_visualizer.h"

namespace coxgraph {

class CoxgraphServer {
 public:
  struct Config {
    int32_t client_number = 0;
    int32_t map_fusion_queue_size = 10;
    ros::Duration refuse_interval = ros::Duration(2);
    int32_t fixed_map_client_id = 0;
    std::string map_frame_prefix = "map";
    std::string output_map_frame = "mission";
    bool enable_registration_constraints = true;
    bool enable_map_fusion_constraints = true;
    int32_t publisher_queue_length = 100;
    bool use_tf_submap_pose = false;
    bool enable_client_loop_closure = false;
    bool publish_global_mesh_on_update = true;

    friend inline std::ostream& operator<<(std::ostream& s, const Config& v) {
      s << std::endl
        << "Coxgraph Server using Config:" << std::endl
        << "  Client Number: " << v.client_number << std::endl
        << "  Map Fusion Queue Size: " << v.map_fusion_queue_size << std::endl
        << "  Client Map Refusion Interval: " << v.refuse_interval << " s"
        << std::endl
        << "  Map Fixed for Client Id: " << v.fixed_map_client_id << std::endl
        << "  Output Mission Frame: " << v.output_map_frame << std::endl
        << "  Registration Constraint: "
        << static_cast<std::string>(
               v.enable_registration_constraints ? "enabled" : "disabled")
        << std::endl
        << "  Map Fusion Constraint: "
        << static_cast<std::string>(
               v.enable_map_fusion_constraints ? "enabled" : "disabled")
        << std::endl
        << "  Client Loop Closure: "
        << static_cast<std::string>(v.enable_client_loop_closure ? "enabled"
                                                                 : "disabled")
        << std::endl
        << "  Publisher Queue Length: " << v.publisher_queue_length << std::endl
        << "  Use TF Submap Pose: "
        << static_cast<std::string>(v.use_tf_submap_pose ? "enabled"
                                                         : "disabled")
        << std::endl
        << "  publish_global_mesh_on_update" << v.publish_global_mesh_on_update
        << std::endl
        << "-------------------------------------------" << std::endl;
      return (s);
    }
  };

  static Config getConfigFromRosParam(const ros::NodeHandle& nh_private);

  CoxgraphServer(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
      : CoxgraphServer(
            nh, nh_private, getConfigFromRosParam(nh_private),
            voxgraph::getVoxgraphSubmapConfigFromRosParams(nh_private),
            voxblox::getMeshIntegratorConfigFromRosParam(nh_private)) {}

  CoxgraphServer(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private,
                 const Config& config, const CliSmConfig& submap_config,
                 const MeshIntegratorConfig& mesh_config)
      : nh_(nh),
        nh_private_(nh_private),
        verbose_(false),
        config_(config),
        submap_config_(submap_config),
        submap_collection_ptr_(std::make_shared<SubmapCollection>(
            submap_config_, config.client_number)),
        pose_graph_interface_(nh_private, submap_collection_ptr_, mesh_config,
                              config.output_map_frame, false),
        server_vis_(
            new ServerVisualizer(nh, nh_private, submap_config, mesh_config)),
        global_mesh_initialized_(false),
        global_mesh_need_update_(0) {
    nh_private_.param<bool>("verbose", verbose_, verbose_);
    LOG(INFO) << "Verbose: " << verbose_;
    LOG(INFO) << config_;

    distrib_ctl_ptr_.reset(
        new DistributionController(nh_, nh_private_, submap_collection_ptr_));

    tf_controller_.reset(new GlobalTfController(
        nh_, nh_private_, config_.client_number, config_.map_frame_prefix,
        distrib_ctl_ptr_, verbose_));
    pose_graph_interface_.setVerbosity(verbose_);
    pose_graph_interface_.setMeasurementConfigFromRosParams(nh_private_);

    subscribeTopics();
    advertiseTopics();
    advertiseServices();
    initClientHandlers(nh_, nh_private_);

    LOG(INFO) << client_handlers_[0]->getConfig();

    CHECK_EQ(config_.fixed_map_client_id, 0)
        << "Fixed map client id has to be set 0 now, since pose graph "
           "optimization set pose of submap 0 as constant";

    if (config_.publish_global_mesh_on_update)
      generate_global_mesh_timer_ = nh_private_.createTimer(
          ros::Duration(1), &CoxgraphServer::generateGlobalMeshEvent, this);
  }

  ~CoxgraphServer() = default;

  void subscribeTopics();
  void advertiseTopics();
  void advertiseServices();

  void mapFusionMsgCallback(const coxgraph_msgs::MapFusion& map_fusion_msg);

  bool getFinalGlobalMeshCallback(
      coxgraph_msgs::FilePath::Request& request,     // NOLINT
      coxgraph_msgs::FilePath::Response& response);  // NOLINT

  bool getPoseHistoryCallback(
      coxgraph_msgs::FilePath::Request& request,     // NOLINT
      coxgraph_msgs::FilePath::Response& response);  // NOLINT

  bool needToFuseCallback(
      coxgraph_msgs::NeedToFuseSrv::Request& request,     // NOLINT
      coxgraph_msgs::NeedToFuseSrv::Response& response);  // NOLINT

  void futureMFProcCallback(const ros::TimerEvent& event);

 private:
  using ClientHandler = server::ClientHandler;
  using GlobalTfController = server::GlobalTfController;
  using ReqState = ClientHandler::ReqState;
  using SubmapCollection = server::SubmapCollection;
  using PoseGraphInterface = server::PoseGraphInterface;
  using ThreadingHelper = voxgraph::ThreadingHelper;
  using PoseMap = PoseGraphInterface::PoseMap;
  using ServerVisualizer = server::ServerVisualizer;
  using DistributionController = server::DistributionController;

  void initClientHandlers(const ros::NodeHandle& nh,
                          const ros::NodeHandle& nh_private);

  void loopClosureCallback(const CliId& client_id,
                           const voxgraph_msgs::LoopClosure& loop_closure_msg);
  bool mapFusionCallback(const coxgraph_msgs::MapFusion& map_fusion_msg,
                         bool future = false);
  void addToMFFuture(const coxgraph_msgs::MapFusion& map_fusion_msg);
  void processMFFuture();
  void timeLineUpdateCallback() {
    processMFFuture();
    global_mesh_need_update_++;
  }
  bool needRefuse(const CliId& cid_a, const ros::Time& time_a,
                  const CliId& cid_b, const ros::Time& time_b);
  bool updateNeedRefuse(const CliId& cid_a, const ros::Time& time_a,
                        const CliId& cid_b, const ros::Time& time_b);

  // TODO(mikexyl): make a pack struct for these vars
  bool fuseMap(const CliId& cid_a, const ros::Time& time_a,
               const CliSmId& cli_sm_id_a, const CliSm::Ptr& submap_a,
               const Transformation& T_submap_t_a, const CliId& cid_b,
               const ros::Time& time_b, const CliSmId& cli_sm_id_b,
               const CliSm::Ptr& submap_b, const Transformation& T_submap_t_b,
               const Transformation& T_t1_t2);

  void updateSubmapRPConstraints();

  enum OptState { FAILED = 0, OK, SKIPPED };
  OptState optimizePoseGraph(bool enable_registration);

  void evaluateResiduals();

  void updateCliMapRelativePose();

  inline bool isTimeFused(const CliId& cid, const ros::Time& time) {
    return fused_time_line_[cid].hasTime(time);
  }
  inline bool isTimeNeedRefuse(const CliId& cid, const ros::Time& time) {
    if (config_.refuse_interval == ros::Duration(0))
      return !isTimeFused(cid, time);
    if (isTimeFused(cid, time)) return false;
    if (time < fused_time_line_[cid].start) {
      return false;
    } else if (time > fused_time_line_[cid].end) {
      return (time - fused_time_line_[cid].end) > config_.refuse_interval;
    }
    return false;
  }

  inline SerSmId addSubmap(const CliSm::Ptr& submap, const CliId& cid,
                           const CliSmId& cli_sm_id) {
    std::lock_guard<std::mutex> submap_add_lock(submap_add_mutex_);
    submap_collection_ptr_->addSubmap(submap, cid, cli_sm_id);
    pose_graph_interface_.addSubmap(submap->getID());
    return submap->getID();
  }

  // Node handles
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  ros::Subscriber map_fusion_sub_;

  // Verbosity and debug mode
  bool verbose_;

  const Config config_;

  const CliSmConfig submap_config_;
  SubmapCollection::Ptr submap_collection_ptr_;
  PoseGraphInterface pose_graph_interface_;
  std::mutex submap_add_mutex_;

  std::vector<ClientHandler::Ptr> client_handlers_;

  // Map fusion msg process related
  std::deque<std::pair<coxgraph_msgs::MapFusion, int>> map_fusion_msgs_future_;
  std::vector<bool> force_fuse_;
  std::vector<TimeLine> fused_time_line_;
  std::map<SerSmId, SerSmId> fused_ser_sm_id_pair;
  std::mutex map_fuse_mutex_;
  std::mutex future_mf_queue_mutex_;
  ros::Timer future_msg_proc_timer_;

  // Asynchronous handle for the pose graph optimization thread
  std::future<OptState> optimization_async_handle_;

  GlobalTfController::Ptr tf_controller_;

  // Visualization
  ServerVisualizer::Ptr server_vis_;
  ros::ServiceServer get_final_global_mesh_srv_;
  ros::ServiceServer get_pose_history_srv_;
  ros::ServiceServer need_to_fuse_srv_;
  std::timed_mutex final_mesh_gen_mutex_;

  DistributionController::Ptr distrib_ctl_ptr_;
  inline bool inControl() const { return distrib_ctl_ptr_->inControl(); }

  ros::Timer generate_global_mesh_timer_;
  int global_mesh_need_update_;
  bool global_mesh_initialized_;
  void generateGlobalMeshEvent(const ros::TimerEvent& /*event*/) {
    if (config_.publish_global_mesh_on_update && global_mesh_initialized_ &&
        global_mesh_need_update_ / config_.client_number == 4) {
      coxgraph_msgs::FilePath file_path_srv;
      file_path_srv.request.file_path = "";
      getFinalGlobalMeshCallback(file_path_srv.request, file_path_srv.response);
      global_mesh_need_update_ = 0;
    }
  }

  constexpr static uint8_t kMaxClientNum = 3;
  constexpr static uint8_t kPoseUpdateWaitMs = 100;
  constexpr static float kFutureMFProcInterval = 1.0;
  constexpr static int kMaxFutureUncatchedN = 4;
};

}  // namespace coxgraph

#endif  // COXGRAPH_SERVER_COXGRAPH_SERVER_H_
