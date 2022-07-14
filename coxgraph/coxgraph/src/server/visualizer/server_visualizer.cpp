#include "coxgraph/server/visualizer/server_visualizer.h"

// #include <open3d/geometry/LineSet.h>
// #include <open3d/io/LineSetIO.h>
// #include <open3d/io/TriangleMeshIO.h>
// #include <open3d/visualization/utility/DrawGeometry.h>

#include <Open3D/Geometry/LineSet.h>
#include <Open3D/IO/ClassIO/LineSetIO.h>
#include <Open3D/IO/ClassIO/TriangleMeshIO.h>
#include <Open3D/Visualization/Utility/DrawGeometry.h>

#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "coxgraph/common.h"
#include "coxgraph/server/submap_collection.h"
#include "coxgraph/utils/msg_converter.h"

namespace coxgraph {
namespace server {
void ServerVisualizer::getFinalGlobalMesh(
    const SubmapCollection::Ptr& submap_collection_ptr,
    const PoseGraphInterface& pose_graph_interface,
    const std::vector<CliSmPack>& other_submaps,
    const std::string& mission_frame, const ros::Publisher& publisher,
    const std::string& file_path, bool save_to_file) {
  LOG(INFO) << "Generating final mesh";

  SubmapCollection::Ptr global_submap_collection_ptr(
      new SubmapCollection(*submap_collection_ptr));
  PoseGraphInterface global_pg_interface(pose_graph_interface,
                                         global_submap_collection_ptr);

  for (auto const& submap_pack : other_submaps) {
    global_submap_collection_ptr->addSubmap(
        submap_pack.submap_ptr, submap_pack.cid, submap_pack.cli_sm_id);
    global_pg_interface.addSubmap(submap_pack.submap_ptr->getID());
  }
  if (global_submap_collection_ptr->getSubmapConstPtrs().empty()) return;

  global_pg_interface.updateSubmapRPConstraints();

  auto opt_async =
      std::async(std::launch::async, &PoseGraphInterface::optimize,
                 &global_pg_interface, config_.registration_enable);

  while (opt_async.wait_for(std::chrono::milliseconds(100)) !=
         std::future_status::ready) {
    LOG_EVERY_N(INFO, 10) << "Global optimzation is still running...";
  }
  global_pg_interface.printResiduals(
      PoseGraphInterface::ConstraintType::RelPose);
  global_pg_interface.printResiduals(
      PoseGraphInterface::ConstraintType::SubmapRelPose);
  LOG(INFO) << "Optimization finished, generating global mesh...";

  global_pg_interface.updateSubmapCollectionPoses();

  auto pose_map = global_pg_interface.getPoseMap();

  boost::filesystem::path mesh_p_o3d_client_color(file_path);
  boost::filesystem::path mesh_p_o3d_raw(file_path);
  boost::filesystem::path mesh_p_voxblox(file_path);
  mesh_p_o3d_client_color.append("global_mesh_o3d_client.ply");
  mesh_p_o3d_raw.append("global_mesh_o3d_raw.ply");
  mesh_p_voxblox.append("global_mesh_voxblox.ply");

  if (config_.o3d_visualize) {
    // Combine mesh
    std::shared_ptr<open3d::geometry::TriangleMesh> combined_mesh(
        new open3d::geometry::TriangleMesh());
    for (auto const& submap :
         global_submap_collection_ptr->getSubmapConstPtrs()) {
      auto submap_mesh = utils::o3dMeshFromMsg(
          *submap->mesh_pointcloud_, config_.o3d_color_mode,
          global_submap_collection_ptr->getCliIdPairBySsid(submap->getID())
              .first);
      if (submap_mesh == nullptr) continue;
      submap_mesh->Transform(
          pose_map[submap->getID()].cast<double>().getTransformationMatrix());
      *combined_mesh += *submap_mesh;
      combined_mesh->MergeCloseVertices(0.06);
      combined_mesh->RemoveDuplicatedVertices();
      combined_mesh->RemoveDuplicatedTriangles();
      combined_mesh->FilterSmoothTaubin(100);
      combined_mesh->SimplifyVertexClustering(0.05);
    }
    //combined_mesh->ComputeVertexNormals();
    //combined_mesh->ComputeTriangleNormals();

    o3d_vis_->ClearGeometries();
    if (config_.o3d_vis_traj) {
      for (int cid = 0; cid < global_submap_collection_ptr->getClientNumber();
           cid++) {
        std::shared_ptr<open3d::geometry::LineSet> traj_line_set(
            new open3d::geometry::LineSet());
        auto traj = global_submap_collection_ptr->getPoseHistory(cid);
        for (size_t i = 0; i < traj.size(); i++) {
          auto const& pose = traj[i];
          traj_line_set->points_.emplace_back(
              pose.pose.position.x, pose.pose.position.y, pose.pose.position.y);
          traj_line_set->colors_.emplace_back(client_colors_[cid]);
          traj_line_set->lines_.emplace_back(i, i + 1);
        }
        traj_line_set->lines_.erase(traj_line_set->lines_.end() - 1);
        o3d_vis_->AddGeometry(traj_line_set);
        o3d_vis_->UpdateGeometry(traj_line_set);
        if (save_to_file) {
          boost::filesystem::path p(file_path);
          p.append("o3d_traj_" + std::to_string(static_cast<int>(cid)) +
                   ".ply");
          open3d::io::WriteLineSetToPLY(p.string(), *traj_line_set);
        }
      }
    }

    o3d_vis_->AddGeometry(combined_mesh);
    o3d_vis_->UpdateGeometry(combined_mesh);
    if (save_to_file)
      open3d::io::WriteTriangleMesh(mesh_p_o3d_client_color.string(),
                                    *combined_mesh);
  }

  if (config_.publish_combined_mesh)
    submap_vis_.saveAndPubCombinedMesh(
        *global_submap_collection_ptr, mission_frame, publisher,
        save_to_file ? mesh_p_voxblox.string() : "");

  std::map<SerSmId, CIdCSIdPair> sm_cli_ids;
  for (auto const& submap : global_submap_collection_ptr->getSubmapPtrs()) {
    auto csid_pair =
        global_submap_collection_ptr->getCliIdPairBySsid(submap->getID());
    sm_cli_ids.emplace(submap->getID(),
                       std::make_pair(csid_pair.first, csid_pair.second));
  }

  LOG(INFO) << "Global mesh generated, published and saved to " << file_path;

  if (save_to_file)
    global_submap_collection_ptr->savePoseHistoryToFile(file_path);

  LOG(INFO) << "Trajectory saved to " << file_path;
}

}  // namespace server
}  // namespace coxgraph
