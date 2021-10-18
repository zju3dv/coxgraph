#include "coxgraph/server/client_handler.h"

#include <coxgraph_msgs/PoseHistorySrv.h>
#include <coxgraph_msgs/SubmapsSrv.h>
#include <coxgraph_msgs/TimeLine.h>

#include <string>
#include <vector>

#include "coxgraph/utils/msg_converter.h"

namespace coxgraph {
namespace server {

ClientHandler::Config ClientHandler::getConfigFromRosParam(
    const ros::NodeHandle& nh_private) {
  ClientHandler::Config config;
  nh_private.param<std::string>("client_handler/client_name_prefix",
                                config.client_name_prefix,
                                config.client_name_prefix);
  nh_private.param<std::string>(
      "client_handler/client_loop_closure_topic_suffix",
      config.client_loop_closure_topic_suffix,
      config.client_loop_closure_topic_suffix);
  nh_private.param<std::string>(
      "client_handler/client_map_pose_update_topic_suffix",
      config.client_map_pose_update_topic_suffix,
      config.client_map_pose_update_topic_suffix);
  nh_private.param<int>("client_handler/pub_queue_length",
                        config.pub_queue_length, config.pub_queue_length);
  nh_private.param<bool>("enable_client_loop_closure",
                         config.enable_client_loop_closure,
                         config.enable_client_loop_closure);
  return config;
}

void ClientHandler::subscribeToTopics() {
  time_line_sub_ =
      nh_.subscribe(client_node_name_ + "/time_line", kSubQueueSize,
                    &ClientHandler::timeLineCallback, this);
  submap_mesh_sub_ =
      nh_.subscribe(client_node_name_ + "/submap_mesh_with_traj", kSubQueueSize,
                    &ClientHandler::submapMeshCallback, this);
  sm_pose_updates_sub_ =
      nh_.subscribe(client_node_name_ + "/map_pose_updates", 10,
                    &ClientHandler::submapPoseUpdatesCallback, this);
}

void ClientHandler::timeLineCallback(
    const coxgraph_msgs::TimeLine& time_line_msg) {
  updateTimeLine(time_line_msg.start, time_line_msg.end);
  time_line_update_callback_();
}

void ClientHandler::advertiseTopics() {
  if (config_.enable_client_loop_closure)
    loop_closure_pub_ = nh_.advertise<voxgraph_msgs::LoopClosure>(
        client_node_name_ + "/" + config_.client_loop_closure_topic_suffix,
        config_.pub_queue_length, true);
  sm_pose_tf_pub_ = nh_.advertise<coxgraph_msgs::MapTransform>(
      client_node_name_ + "/" + config_.client_map_pose_update_topic_suffix,
      config_.pub_queue_length, true);
}

void ClientHandler::subscribeToServices() {
  LOG(INFO) << log_prefix_ << "Subscribed to service: "
            << client_node_name_ + "/get_client_submap";
  pub_client_submap_client_ = nh_.serviceClient<coxgraph_msgs::ClientSubmapSrv>(
      client_node_name_ + "/get_client_submap");

  LOG(INFO) << log_prefix_ << "Subscribed to service: "
            << client_node_name_ + "/get_all_submaps";
  get_all_submaps_client_ = nh_.serviceClient<coxgraph_msgs::SubmapsSrv>(
      client_node_name_ + "/get_all_submaps");

  LOG(INFO) << log_prefix_ << "Subscribed to service: "
            << client_node_name_ + "/get_pose_history";
  get_pose_history_client_ = nh_.serviceClient<coxgraph_msgs::PoseHistorySrv>(
      client_node_name_ + "/get_pose_history");
}

ClientHandler::ReqState ClientHandler::requestSubmapByTime(
    const ros::Time& timestamp, const SerSmId& ser_sid, CliSmId* cli_sid,
    CliSm::Ptr* submap, Transformation* T_Sm_C_t) {
  std::lock_guard<std::mutex> submap_request_lock(submap_request_mutex_);

  if (!time_line_.hasTime(timestamp)) return ReqState::FUTURE;

  coxgraph_msgs::ClientSubmapSrv cli_submap_srv;
  cli_submap_srv.request.timestamp = timestamp;
  if (pub_client_submap_client_.call(cli_submap_srv)) {
    eval_data_pub_.publishBandwidth(
        client_node_name_ + "/client_submap",
        utils::sizeOfMsg(cli_submap_srv.response.submap),
        cli_submap_srv.response.pub_time, ros::Time::now());
    *cli_sid = cli_submap_srv.response.submap.map_header.id;
    *submap = utils::cliSubmapFromMsg(ser_sid, submap_config_,
                                      cli_submap_srv.response, &map_frame_id_);
    tf::transformMsgToKindr<voxblox::FloatingPoint>(
        cli_submap_srv.response.transform, T_Sm_C_t);
    return ReqState::SUCCESS;
  }
  return ReqState::FAILED;
}

void ClientHandler::submapPoseUpdatesCallback(
    const coxgraph_msgs::MapPoseUpdates& map_pose_updates_msg) {
  CHECK(submap_collection_ptr_ != nullptr);
  LOG(INFO) << log_prefix_ << "Received new pose for "
            << map_pose_updates_msg.submap_id.size() << " submaps.";
  for (int i = 0; i < map_pose_updates_msg.submap_id.size(); i++) {
    SerSmId ser_sm_id;
    CHECK(submap_collection_ptr_->getSerSmIdByCliSmId(
        client_id_, map_pose_updates_msg.submap_id[i], &ser_sm_id));
    CHECK(submap_collection_ptr_->exists(ser_sm_id))
        << "CliSmId " << map_pose_updates_msg.submap_id[i]
        << ", SerSmId: " << ser_sm_id;
    geometry_msgs::Pose submap_pose_msg = map_pose_updates_msg.new_pose[i];
    CliSm::Ptr submap_ptr = submap_collection_ptr_->getSubmapPtr(ser_sm_id);
    TransformationD submap_pose;
    tf::poseMsgToKindr(submap_pose_msg, &submap_pose);
    submap_ptr = submap_collection_ptr_->getSubmapPtr(ser_sm_id);
    submap_ptr->setPose(submap_pose.cast<voxblox::FloatingPoint>());
    submap_collection_ptr_->updateOriPose(
        ser_sm_id, submap_pose.cast<voxblox::FloatingPoint>());
    LOG(INFO) << log_prefix_ << "Updating pose for submap cli id: "
              << map_pose_updates_msg.submap_id[i] << " ser id: " << ser_sm_id;
  }
}

bool ClientHandler::requestAllSubmaps(std::vector<CliSmPack>* submap_packs,
                                      SerSmId* start_ser_sm_id) {
  std::lock_guard<std::mutex> submap_request_lock(submap_request_mutex_);

  CHECK(submap_packs != nullptr);
  submap_packs->clear();
  coxgraph_msgs::SubmapsSrv submap_srv;
  if (get_all_submaps_client_.call(submap_srv)) {
    for (auto const& submap_msg : submap_srv.response.submaps)
      submap_packs->emplace_back(
          utils::cliSubmapFromMsg((*start_ser_sm_id)++, submap_config_,
                                  submap_msg, &map_frame_id_),
          client_id_, submap_msg.map_header.id);
    return true;
  }
  return false;
}

bool ClientHandler::requestPoseHistory(const std::string& file_path,
                                       PoseStampedVector* pose_history) {
  coxgraph_msgs::PoseHistorySrv pose_history_srv;
  pose_history_srv.request.file_path = file_path;
  CHECK(pose_history != nullptr);
  if (get_pose_history_client_.call(pose_history_srv)) {
    *pose_history = pose_history_srv.response.pose_history.pose_history;
    return true;
  } else {
    return false;
  }
}

bool ClientHandler::lookUpSubmapPoseFromTf(CliSmId sid,
                                           Transformation* T_Cli_Sm) {
  CHECK(T_Cli_Sm != nullptr);
  std::string submap_pose_frame =
      "submap_" + std::to_string(sid) + "_" + std::to_string(client_id_);
  ros::Time zero_timestamp = ros::Time(0);
  if (transformer_.lookupTransform(map_frame_id_, submap_pose_frame,
                                   zero_timestamp, T_Cli_Sm))
    return true;
  else
    LOG(WARNING) << "Failed to look up Tf from " << map_frame_id_ << " to "
                 << submap_pose_frame << " at time " << zero_timestamp;
  return false;
}

}  // namespace server
}  // namespace coxgraph
