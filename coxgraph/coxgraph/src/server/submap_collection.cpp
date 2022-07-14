#include "coxgraph/server/submap_collection.h"

#include <voxblox/integrator/merge_integration.h>

#include <vector>

namespace coxgraph {
namespace server {

Transformation SubmapCollection::addSubmap(const CliSm::Ptr& submap_ptr,
                                           const CliId& cid,
                                           const CliSmId& cli_sm_id) {
  CHECK(submap_ptr != nullptr);
  voxgraph::VoxgraphSubmapCollection::addSubmap(submap_ptr);
  sm_cli_id_map_.emplace(submap_ptr->getID(), CIdCSIdPair(cid, cli_sm_id));
  if (!cli_ser_sm_id_map_.count(submap_ptr->getID())) {
    cli_ser_sm_id_map_.emplace(cid, std::vector<SerSmId>());
  }
  cli_ser_sm_id_map_[cid].emplace_back(submap_ptr->getID());
  sm_id_ori_pose_map_.emplace(submap_ptr->getID(), submap_ptr->getPose());
  return Transformation();
}

Transformation SubmapCollection::mergeToCliMap(const CliSm::Ptr& submap_ptr) {
  CHECK(exists(submap_ptr->getID()));

  auto const& cli_map_ptr = getSubmapPtr(submap_ptr->getID());
  // TODO(mikexyl): only merge layers now, merge more if needed, and
  // theoretically not need to transform layer since it's already done when
  // generating submap from msg
  voxblox::mergeLayerAintoLayerB(
      submap_ptr->getTsdfMapPtr()->getTsdfLayer(),
      cli_map_ptr->getTsdfMapPtr()->getTsdfLayerPtr());

  cli_map_ptr->finishSubmap();
  return submap_ptr->getPose() * cli_map_ptr->getPose().inverse();
}

}  // namespace server
}  // namespace coxgraph
