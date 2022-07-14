#ifndef COXGRAPH_COMMON_H_
#define COXGRAPH_COMMON_H_

#include <cblox/core/common.h>
#include <geometry_msgs/PoseStamped.h>
#include <voxblox/core/common.h>
#include <voxblox/mesh/mesh_integrator.h>
#include <voxblox_ros/transformer.h>
#include <voxgraph/backend/constraint/constraint.h>
#include <voxgraph/common.h>
#include <voxgraph/frontend/frame_names.h>
#include <voxgraph/frontend/submap_collection/bounding_box.h>
#include <voxgraph/frontend/submap_collection/voxgraph_submap.h>
#include <voxgraph/tools/tf_helper.h>

#include <utility>
#include <vector>

namespace coxgraph {

typedef int8_t CliId;

using CliSm = voxgraph::VoxgraphSubmap;
using SerSmId = voxgraph::SubmapID;
using CliSmId = voxgraph::SubmapID;
typedef std::pair<CliId, CliSmId> CIdCSIdPair;

struct CliSmPack {
  CliSmPack(const CliSm::Ptr& submap_ptr_in, const CliId& cid_in,
            const CliSmId& cli_sm_id_in)
      : submap_ptr(submap_ptr_in), cid(cid_in), cli_sm_id(cli_sm_id_in) {}
  CliSm::Ptr submap_ptr;
  CliId cid;
  CliSmId cli_sm_id;
};

using CliSmConfig = voxgraph::VoxgraphSubmap::Config;
using MeshIntegratorConfig = voxblox::MeshIntegratorConfig;

using Transformation = voxgraph::Transformation;
using TransformationD = voxgraph::TransformationD;
using TransformationVector = cblox::TransformationVector;
using InformationMatrix = voxgraph::Constraint::InformationMatrix;

using FrameNames = voxgraph::FrameNames;
using BoundingBox = voxgraph::BoundingBox;
using TfHelper = voxgraph::TfHelper;
using Transformer = voxblox::Transformer;

struct TimeLine {
  TimeLine() : start(0), end(0) {}
  ros::Time start;
  ros::Time end;
  bool hasTime(const ros::Time& time) {
    if (end.isZero()) return false;
    if (time >= start && time <= end) {
      return true;
    }
    return false;
  }
  bool update(const ros::Time& new_start, const ros::Time& new_end) {
    if (start != new_start || end != new_end) {
      start = new_start;
      end = new_end;
      return true;
    }
    return false;
  }
  bool update(const ros::Time& new_time) {
    if (new_time < start) {
      start = new_time;
      return true;
    } else if (new_time > end) {
      end = new_time;
      return true;
    }
    return false;
  }
  bool setEnd(const ros::Time& new_end) {
    if (end == new_end) return false;
    end = new_end;
    return true;
  }
};

typedef std::function<void()> TimeLineUpdateCallback;

typedef std::vector<geometry_msgs::PoseStamped> PoseStampedVector;

}  // namespace coxgraph

#endif  // COXGRAPH_COMMON_H_
