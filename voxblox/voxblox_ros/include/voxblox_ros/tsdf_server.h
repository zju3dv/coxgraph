#ifndef VOXBLOX_ROS_TSDF_SERVER_H_
#define VOXBLOX_ROS_TSDF_SERVER_H_

#include <deque>
#include <memory>
#include <queue>
#include <string>

#include <minkindr_conversions/kindr_msg.h>
#include <pcl/conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_srvs/Empty.h>
#include <std_srvs/SetBool.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>

#include <voxblox/alignment/icp.h>
#include <voxblox/core/tsdf_map.h>
#include <voxblox/integrator/merge_integration.h>
#include <voxblox/integrator/tsdf_integrator.h>
#include <voxblox/io/layer_io.h>
#include <voxblox/io/mesh_ply.h>
#include <voxblox/mesh/mesh_integrator.h>
#include <voxblox/utils/color_maps.h>
#include <voxblox_msgs/FilePath.h>
#include <voxblox_msgs/Mesh.h>

#include "voxblox_ros/mesh_vis.h"
#include "voxblox_ros/ptcloud_vis.h"
#include "voxblox_ros/transformer.h"

namespace voxblox {

constexpr float kDefaultMaxIntensity = 100.0;

class TsdfServer {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TsdfServer(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);
  TsdfServer(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private,
             const TsdfMap::Config& config,
             const TsdfIntegratorBase::Config& integrator_config,
             const MeshIntegratorConfig& mesh_config);
  virtual ~TsdfServer() {}

  void getServerConfigFromRosParam(const ros::NodeHandle& nh_private);

  void insertPointcloud(const sensor_msgs::PointCloud2::Ptr& pointcloud);

  void insertFreespacePointcloud(
      const sensor_msgs::PointCloud2::Ptr& pointcloud);

  virtual void processPointCloudMessageAndInsert(
      const sensor_msgs::PointCloud2::Ptr& pointcloud_msg,
      const Transformation& T_G_C, const bool is_freespace_pointcloud);

  void integratePointcloud(const ros::Time& timestamp,
                           const Transformation& T_G_C,
                           std::shared_ptr<const Pointcloud> ptcloud_C,
                           std::shared_ptr<const Colors> colors,
                           const bool is_freespace_pointcloud = false);

  // Note(schmluk): Provide the legacy interface for voxblox-specific packages.
  void integratePointcloud(const Transformation& T_G_C,
                           const Pointcloud& ptcloud_C, const Colors& colors,
                           const bool is_freespace_pointcloud = false);

  void servicePointcloudDeintegrationQueue();

  virtual void newPoseCallback(const Transformation& T_G_C) {
    if (slice_level_follow_robot_) {
      slice_level_ = T_G_C.getPosition().z();
    }
  }

  void publishAllUpdatedTsdfVoxels();
  void publishTsdfSurfacePoints();
  void publishTsdfOccupiedNodes();

  virtual void publishSlices();
  /// Incremental update.
  virtual void updateMesh();
  /// Batch update.
  virtual bool generateMesh();
  // Publishes all available pointclouds.
  virtual void publishPointclouds();
  // Publishes the complete map
  virtual void publishMap(bool reset_remote_map = false);
  virtual bool saveMap(const std::string& file_path);
  virtual bool loadMap(const std::string& file_path);

  bool clearMapCallback(std_srvs::Empty::Request& request,           // NOLINT
                        std_srvs::Empty::Response& response);        // NOLINT
  bool saveMapCallback(voxblox_msgs::FilePath::Request& request,     // NOLINT
                       voxblox_msgs::FilePath::Response& response);  // NOLINT
  bool loadMapCallback(voxblox_msgs::FilePath::Request& request,     // NOLINT
                       voxblox_msgs::FilePath::Response& response);  // NOLINT
  bool generateMeshCallback(std_srvs::Empty::Request& request,       // NOLINT
                            std_srvs::Empty::Response& response);    // NOLINT
  bool publishPointcloudsCallback(
      std_srvs::Empty::Request& request,                             // NOLINT
      std_srvs::Empty::Response& response);                          // NOLINT
  bool publishTsdfMapCallback(std_srvs::Empty::Request& request,     // NOLINT
                              std_srvs::Empty::Response& response);  // NOLINT

  void updateMeshEvent(const ros::TimerEvent& event);
  void publishMapEvent(const ros::TimerEvent& event);

  std::shared_ptr<TsdfMap> getTsdfMapPtr() { return tsdf_map_; }
  std::shared_ptr<const TsdfMap> getTsdfMapPtr() const { return tsdf_map_; }

  /// Accessors for setting and getting parameters.
  double getSliceLevel() const { return slice_level_; }
  void setSliceLevel(double slice_level) { slice_level_ = slice_level; }

  bool setPublishSlices() const { return publish_slices_; }
  void setPublishSlices(const bool publish_slices) {
    publish_slices_ = publish_slices;
  }

  void setWorldFrame(const std::string& world_frame) {
    world_frame_ = world_frame;
  }
  std::string getWorldFrame() const { return world_frame_; }

  /// CLEARS THE ENTIRE MAP!
  virtual void clear();

  /// Overwrites the layer with what's coming from the topic!
  void tsdfMapCallback(const voxblox_msgs::Layer& layer_msg);

  static Transformation gravityAlignPose(const Transformation& input_pose) {
    // Use the logarithmic map to get the pose's [x, y, z, r, p, y] components
    Transformation::Vector6 T_vec = input_pose.log();

    // Set the roll and pitch to zero
    T_vec[3] = 0;
    T_vec[4] = 0;

    // Return the gravity aligned pose as a translation + quaternion,
    // using the exponential map
    return Transformation::exp(T_vec);
  }

 protected:
  /**
   * Gets the next pointcloud that has an available transform to process from
   * the queue.
   */
  bool getNextPointcloudFromQueue(
      std::queue<sensor_msgs::PointCloud2::Ptr>* queue,
      sensor_msgs::PointCloud2::Ptr* pointcloud_msg, Transformation* T_G_C);

  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  /// Data subscribers.
  ros::Subscriber pointcloud_sub_;
  ros::Subscriber freespace_pointcloud_sub_;

  /// Publish markers for visualization.
  ros::Publisher mesh_pub_;
  ros::Publisher tsdf_pointcloud_pub_;
  ros::Publisher surface_pointcloud_pub_;
  ros::Publisher tsdf_slice_pub_;
  ros::Publisher occupancy_marker_pub_;
  ros::Publisher icp_transform_pub_;
  ros::Publisher reprojected_pointcloud_pub_;
  ros::Publisher mesh_with_history_pub_;

  /// Publish the complete map for other nodes to consume.
  ros::Publisher tsdf_map_pub_;

  /// Subscriber to subscribe to another node generating the map.
  ros::Subscriber tsdf_map_sub_;

  // Services.
  ros::ServiceServer generate_mesh_srv_;
  ros::ServiceServer clear_map_srv_;
  ros::ServiceServer save_map_srv_;
  ros::ServiceServer load_map_srv_;
  ros::ServiceServer publish_pointclouds_srv_;
  ros::ServiceServer publish_tsdf_map_srv_;

  /// Tools for broadcasting TFs.
  tf::TransformBroadcaster tf_broadcaster_;

  // Timers.
  ros::Timer update_mesh_timer_;
  ros::Timer publish_map_timer_;

  bool verbose_;

  /**
   * Global/map coordinate frame. Will always look up TF transforms to this
   * frame.
   */
  std::string world_frame_;
  /** * Name of the ICP corrected frame. Publishes TF and transform topic to
   * this * if ICP on.
   */
  std::string icp_corrected_frame_;
  /// Name of the pose in the ICP correct Frame.
  std::string pose_corrected_frame_;

  /// Delete blocks that are far from the system to help manage memory
  double max_block_distance_from_body_;

  /// Pointcloud visualization settings.
  double slice_level_;
  bool slice_level_follow_robot_;

  /// If the system should subscribe to a pointcloud giving points in freespace
  bool use_freespace_pointcloud_;

  /**
   * Mesh output settings. Mesh is only written to file if mesh_filename_ is
   * not empty.
   */
  std::string mesh_filename_;
  /// How to color the mesh.
  ColorMode color_mode_;

  /// Colormap to use for intensity pointclouds.
  std::shared_ptr<ColorMap> color_map_;

  /// Will throttle to this message rate.
  ros::Duration min_time_between_msgs_;

  /// What output information to publish
  bool publish_pointclouds_on_update_;
  bool publish_slices_;
  bool publish_pointclouds_;
  bool publish_tsdf_map_;

  /// Whether to save the latest mesh message sent (for inheriting classes).
  bool cache_mesh_;

  /**
   *Whether to enable ICP corrections. Every pointcloud coming in will attempt
   * to be matched up to the existing structure using ICP. Requires the initial
   * guess from odometry to already be very good.
   */
  bool enable_icp_;
  /**
   * If using ICP corrections, whether to store accumulate the corrected
   * transform. If this is set to false, the transform will reset every
   * iteration.
   */
  bool accumulate_icp_corrections_;

  /// Subscriber settings.
  int pointcloud_queue_size_;
  int num_subscribers_tsdf_map_;

  // Maps and integrators.
  std::shared_ptr<TsdfMap> tsdf_map_;
  TsdfIntegratorBase::Ptr tsdf_integrator_;

  /// ICP matcher
  std::shared_ptr<ICP> icp_;

  // Mesh accessories.
  std::shared_ptr<MeshLayer> mesh_layer_;
  std::unique_ptr<MeshIntegrator<TsdfVoxel>> mesh_integrator_;
  /// Optionally cached mesh message.
  voxblox_msgs::Mesh cached_mesh_msg_;

  /**
   * Transformer object to keep track of either TF transforms or messages from
   * a transform topic.
   */
  Transformer transformer_;
  /**
   * Queue of incoming pointclouds, in case the transforms can't be immediately
   * resolved.
   */
  std::queue<sensor_msgs::PointCloud2::Ptr> pointcloud_queue_;
  std::queue<sensor_msgs::PointCloud2::Ptr> freespace_pointcloud_queue_;

  // TODO(victorr): Add description
  struct PointcloudDeintegrationPacket {
    const ros::Time timestamp;
    Transformation T_G_C;
    std::shared_ptr<const Pointcloud> ptcloud_C;
    std::shared_ptr<const Colors> colors;
    const bool is_freespace_pointcloud;
  };
  size_t pointcloud_deintegration_queue_length_;
  std::deque<PointcloudDeintegrationPacket> pointcloud_deintegration_queue_;
  const size_t num_voxels_per_block_;
  bool map_needs_pruning_;
  virtual void pruneMap();

  // Last message times for throttling input.
  ros::Time last_msg_time_ptcloud_;
  ros::Time last_msg_time_freespace_ptcloud_;

  /// Current transform corrections from ICP.
  Transformation icp_corrected_transform_;

  // TODO(victorr): Add description
  bool publish_map_with_trajectory_;

  float publish_active_tsdf_every_n_sec_;
  ros::Timer active_map_pub_timer_;
  ros::Publisher active_tsdf_pub_;
  int num_subscribers_active_tsdf_;
  virtual void activeMapPubCallback(const ros::TimerEvent&) {
    int tsdf_subscribers = active_tsdf_pub_.getNumSubscribers();
    if (tsdf_subscribers > 0) {
      voxblox_msgs::Layer tsdf_layer_msg;
      serializeLayerAsMsg<TsdfVoxel>(tsdf_map_->getTsdfLayer(), false,
                                     &tsdf_layer_msg);

      tsdf_layer_msg.action = voxblox_msgs::Layer::ACTION_MERGE;
      active_tsdf_pub_.publish(tsdf_layer_msg);
    }
    num_subscribers_active_tsdf_ = tsdf_subscribers;
  }

  bool map_running_;
  ros::ServiceServer toggle_mapping_srv_;
  bool toogleMappingCallback(std_srvs::SetBool::Request& request,      // NOLINT
                             std_srvs::SetBool::Response& response) {  // NOLINT
    map_running_ = request.data;
    response.success = true;
    response.message = "Mapping " + static_cast<std::string>(
                                        map_running_ ? "started" : "stopped");
    return true;
  }

  std::string pointcloud_frame_;

  float submap_interval_;
  ros::Time last_submap_stamp_;
  std::vector<Transformation> pose_history_queue_;

  MeshIntegratorConfig mesh_histroy_config_;

  bool publish_mesh_with_history_ = false;

  ros::Timer create_new_submap_timer_;
  void createNewSubmapEvent(const ros::TimerEvent& /*event*/) {
    createNewSubmap();
  }
  void createNewSubmap() {
    if (submap_interval_ > 0.0 &&
        (last_msg_time_ptcloud_ - last_submap_stamp_).toSec() >
            submap_interval_) {
      // switch to new submap
      last_submap_stamp_ = last_msg_time_ptcloud_;
      publishMap();
      tsdf_map_->getTsdfLayerPtr()->removeAllBlocks();
      pointcloud_deintegration_queue_.clear();
      tsdf_integrator_->resetObsCnt(ros::Time::now().toSec());
    }
  }

  void transformLayerToSubmapFrame() {
    if (pointcloud_deintegration_queue_.empty()) return;
    AlignedVector<Transformation> trajectory;
    for (auto const& pointcloud_packet : pointcloud_deintegration_queue_) {
      trajectory.emplace_back(pointcloud_packet.T_G_C);
    }
    const size_t trajectory_middle_idx = trajectory.size() / 2;
    Transformation T_odom_trajectory_middle_pose =
        trajectory[trajectory_middle_idx];
    const Transformation T_odom_submap = gravityAlignPose(
        T_odom_trajectory_middle_pose.cast<voxblox::FloatingPoint>());
    Layer<TsdfVoxel> old_tsdf_layer(tsdf_map_->getTsdfLayer());
    tsdf_map_->getTsdfLayerPtr()->removeAllBlocks();
    transformLayer(old_tsdf_layer, T_odom_submap.inverse(),
                   tsdf_map_->getTsdfLayerPtr());
  }

  int max_gap_, min_n_;
  void publishMeshWithHistory() {
    std::shared_ptr<MeshLayer> mesh_layer(
        new MeshLayer(tsdf_map_->block_size()));
    mesh_histroy_config_.use_history = true;
    std::shared_ptr<MeshIntegrator<TsdfVoxel>> mesh_integrator(
        new MeshIntegrator<TsdfVoxel>(mesh_histroy_config_,
                                      tsdf_map_->getTsdfLayerPtr(),
                                      mesh_layer.get()));

    mesh_integrator->generateMesh(false, true);
    double start_time =
        pointcloud_deintegration_queue_.begin()->timestamp.toSec();
    int stamp_offset =
        start_time == tsdf_integrator_->obs_time
            ? 0
            : std::round((start_time - tsdf_integrator_->obs_time) / 0.05);

    mesh_integrator->addHistoryToMesh(stamp_offset, max_gap_, min_n_);

    voxblox_msgs::Mesh mesh_msg;
    generateVoxbloxMeshMsg(mesh_layer, color_mode_, &mesh_msg);
    mesh_msg.header.frame_id = world_frame_;

    for (const PointcloudDeintegrationPacket& pointcloud_queue_packet :
         pointcloud_deintegration_queue_) {
      geometry_msgs::PoseStamped pose_msg;
      pose_msg.header.frame_id = world_frame_;
      pose_msg.header.stamp = pointcloud_queue_packet.timestamp;
      tf::poseKindrToMsg(pointcloud_queue_packet.T_G_C.cast<double>(),
                         &pose_msg.pose);
      mesh_msg.trajectory.poses.emplace_back(pose_msg);
    }

    mesh_with_history_pub_.publish(mesh_msg);
  }
};

}  // namespace voxblox

#endif  // VOXBLOX_ROS_TSDF_SERVER_H_
