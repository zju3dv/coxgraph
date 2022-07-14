#ifndef COXGRAPH_CLIENT_COXGRAPH_CLIENT_H_
#define COXGRAPH_CLIENT_COXGRAPH_CLIENT_H_

// #include <open3d/geometry/LineSet.h>
// #include <open3d/visualization/visualizer/RenderOption.h>
// #include <open3d/visualization/visualizer/Visualizer.h>
#include <Open3D/Geometry/LineSet.h>
#include <Open3D/Visualization/Visualizer/RenderOption.h>
#include <Open3D/Visualization/Visualizer/Visualizer.h>
#include <coxgraph_msgs/ClientSubmap.h>
#include <coxgraph_msgs/ClientSubmapSrv.h>
#include <coxgraph_msgs/PoseHistorySrv.h>
#include <coxgraph_msgs/SubmapsSrv.h>
#include <coxgraph_msgs/TimeLine.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <pcl_ros/transforms.h>
#include <sensor_msgs/PointCloud2.h>
#include <voxgraph/frontend/voxgraph_mapper.h>
#include <voxgraph_msgs/LoopClosure.h>

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include "coxgraph/client/map_server.h"
#include "coxgraph/common.h"
#include "coxgraph/utils/msg_converter.h"

namespace coxgraph {
class CoxgraphClient : public voxgraph::VoxgraphMapper {
 public:
  CoxgraphClient(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
      : VoxgraphMapper(nh, nh_private),
        recover_mode_(true),
        vis_combined_o3d_mesh_(false) {
    int client_id;
    nh_private.param<int>("client_id", client_id, -1);
    client_id_ = static_cast<CliId>(client_id);

    nh_private.param("recover_mode", recover_mode_, recover_mode_);
    nh_private.param("vis_combined_o3d_mesh", vis_combined_o3d_mesh_,
                     vis_combined_o3d_mesh_);
    if (vis_combined_o3d_mesh_) {
      o3d_vis_ = new open3d::visualization::Visualizer();
      o3d_vis_->CreateVisualizerWindow("client_" + std::to_string(client_id_));
      o3d_vis_->GetRenderOption().mesh_color_option_ =
          open3d::visualization::RenderOption::MeshColorOption::Normal;
      traj_color_ = Eigen::Vector3d(0, 1.0, 0);
      // o3d_vis_->AddGeometry(combined_mesh_);
      o3d_mesh_timer_ = nh_private_.createTimer(
          ros::Duration(0.01), &CoxgraphClient::o3dMeshVisualizeEvent, this);
      // o3d_run_thread_ = std::thread([this]() { this->o3d_vis_->Run(); });
    }
    subscribeToClientTopics();
    advertiseClientTopics();
    advertiseClientServices();
    if (client_id_ < 0) {
      LOG(FATAL) << "Invalid Client Id " << client_id_;
    } else {
      LOG(INFO) << "Started Coxgraph Client " << client_id_;
    }
    log_prefix_ = "Client " + std::to_string(client_id_) + ": ";

    map_server_.reset(new MapServer(nh_, nh_private_, client_id_,
                                    submap_config_, frame_names_,
                                    submap_collection_ptr_));
  }
  std::thread o3d_run_thread_;

  ~CoxgraphClient() = default;

  inline const CliId& getClientId() const { return client_id_; }

  void subscribeToClientTopics();
  void advertiseClientTopics();
  void advertiseClientServices();

  bool getClientSubmapCallback(
      coxgraph_msgs::ClientSubmapSrv::Request& request,     // NOLINT
      coxgraph_msgs::ClientSubmapSrv::Response& response);  // NOLINT

  bool getAllClientSubmapsCallback(
      coxgraph_msgs::SubmapsSrv::Request& request,     // NOLINT
      coxgraph_msgs::SubmapsSrv::Response& response);  // NOLINT

  bool getPoseHistory(
      coxgraph_msgs::PoseHistorySrv::Request& request,      // NOLINT
      coxgraph_msgs::PoseHistorySrv::Response& response) {  // NOLINT
    response.pose_history.pose_history =
        submap_collection_ptr_->getPoseHistory();
    boost::filesystem::path p(request.file_path);
    p.append("coxgraph_client_traj_" + std::to_string(client_id_) + ".txt");
    savePoseHistory(p.string());
    return true;
  }

  bool submapCallback(const voxblox_msgs::LayerWithTrajectory& submap_msg,
                      bool transform_layer) override;

 private:
  using VoxgraphMapper = voxgraph::VoxgraphMapper;
  using MapServer = client::MapServer;
  typedef std::map<CliSmId, Transformation> SmIdTfMap;

  void publishTimeLine();
  void publishMapPoseUpdates();
  void publishSubmapPoseTFs() override;

  void savePoseHistory(std::string file_path);

  CliId client_id_;
  std::string log_prefix_;

  ros::Publisher time_line_pub_;
  ros::Publisher map_pose_pub_;
  ros::Publisher submap_mesh_pub_;
  ros::ServiceServer get_client_submap_srv_;
  ros::ServiceServer get_all_client_submaps_srv_;
  ros::ServiceServer get_pose_history_srv_;

  SmIdTfMap ser_sm_id_pose_map_;

  std::timed_mutex submap_proc_mutex_;

  MapServer::Ptr map_server_;

  bool recover_mode_;
  typedef message_filters::sync_policies::ApproximateTime<
      voxblox_msgs::LayerWithTrajectory, sensor_msgs::PointCloud2>
      sync_pol;
  message_filters::Synchronizer<sync_pol>* synchronizer_;
  message_filters::Subscriber<voxblox_msgs::LayerWithTrajectory>*
      submap_sync_sub_;
  message_filters::Subscriber<sensor_msgs::PointCloud2>*
      mesh_pointcloud_sync_sub_;

  // T_Submap_Mesh
  std::map<CliSmId, std::pair<TransformationD,
                              std::shared_ptr<open3d::geometry::TriangleMesh>>>
      mesh_collection_;
  Eigen::Vector3d traj_color_;
  open3d::visualization::Visualizer* o3d_vis_;
  void submapMeshCallback(
      const voxblox_msgs::LayerWithTrajectoryConstPtr& layer_msg,
      const sensor_msgs::PointCloud2ConstPtr& pointcloud_msg) {
    if (submapCallback(*layer_msg, true)) {
      auto o3d_mesh = utils::o3dMeshFromMsg(*pointcloud_msg);
      if (o3d_mesh != nullptr) {
        auto T_G_Sm = submap_collection_ptr_->getActiveSubmapPose();
        if (vis_combined_o3d_mesh_) {
          // o3d_mesh: T_G_Mesh
          mesh_collection_.emplace(
              submap_collection_ptr_->getActiveSubmapID(),
              std::make_pair(T_G_Sm.cast<double>(), o3d_mesh));
          // o3d_vis_->AddGeometry(o3d_mesh);
          updateCombinedMesh();
        }

        sensor_msgs::PointCloud2::Ptr T_Sm_P(new sensor_msgs::PointCloud2());
        pcl_ros::transformPointCloud(T_G_Sm.inverse().getTransformationMatrix(),
                                     *pointcloud_msg, *T_Sm_P);
        submap_collection_ptr_->getActiveSubmapPtr()->mesh_pointcloud_ = T_Sm_P;
      }
    }
  }

  bool vis_combined_o3d_mesh_;
  ros::Timer o3d_mesh_timer_;
  void o3dMeshVisualizeEvent(const ros::TimerEvent& /*event*/) {
    o3d_vis_->PollEvents();
    o3d_vis_->UpdateRender();
  }

  void updateCombinedMesh() {
    o3d_vis_->ClearGeometries();

    std::shared_ptr<open3d::geometry::TriangleMesh> combined_mesh(
        new open3d::geometry::TriangleMesh());
    std::shared_ptr<open3d::geometry::LineSet> traj_line_set(
        new open3d::geometry::LineSet());
    for (auto& kv : mesh_collection_) {
      Transformation T_G_new;
      submap_collection_ptr_->getSubmapPose(kv.first, &T_G_new);
      // Transform to T_new_Mesh
      TransformationD T_old_new, T_G_old = kv.second.first;
      T_old_new = T_G_old.inverse() * T_G_new.cast<double>();
      kv.second.first = T_G_new.cast<double>();
      kv.second.second->Transform(T_old_new.getTransformationMatrix());
      *combined_mesh += *kv.second.second;
      combined_mesh->MergeCloseVertices(0.02);
      combined_mesh->RemoveDuplicatedVertices();
      combined_mesh->RemoveDuplicatedTriangles();
      combined_mesh->ComputeVertexNormals();
      combined_mesh->ComputeTriangleNormals();
    }
    o3d_vis_->AddGeometry(combined_mesh);
    o3d_vis_->UpdateGeometry(combined_mesh);

    traj_line_set->Clear();
    auto traj = submap_collection_ptr_->getPoseHistory();
    for (size_t i = 0; i < traj.size(); i++) {
      auto const& pose = traj[i];
      traj_line_set->points_.emplace_back(
          pose.pose.position.x, pose.pose.position.y, pose.pose.position.y);
      traj_line_set->colors_.emplace_back(traj_color_);
      traj_line_set->lines_.emplace_back(i, i + 1);
    }
    traj_line_set->lines_.erase(traj_line_set->lines_.end() - 1);
    o3d_vis_->AddGeometry(traj_line_set);
    o3d_vis_->UpdateGeometry(traj_line_set);
  }
};

}  // namespace coxgraph

#endif  // COXGRAPH_CLIENT_COXGRAPH_CLIENT_H_
