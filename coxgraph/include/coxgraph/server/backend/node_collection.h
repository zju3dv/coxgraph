#ifndef COXGRAPH_SERVER_BACKEND_NODE_COLLECTION_H_
#define COXGRAPH_SERVER_BACKEND_NODE_COLLECTION_H_

#include <voxgraph/backend/local_parameterization/angle_local_parameterization.h>
#include <voxgraph/backend/node/node.h>

#include <map>
#include <memory>

#include "coxgraph/common.h"
#include "coxgraph/server/backend/client_frame_node.h"

namespace coxgraph {

namespace server {

class NodeCollection {
 public:
  typedef std::map<CliId, ClientFrameNode::Ptr> ClientFrameNodeMap;

  NodeCollection() {
    local_parameterization_ = std::make_shared<ceres::ProductParameterization>(
        new ceres::IdentityParameterization(3),
        voxgraph::AngleLocalParameterization::Create());
  }

  void addClientNode(const ClientFrameNode::Config& config) {
    auto client_node_ptr =
        std::make_shared<ClientFrameNode>(newNodeId(), config);
    client_frame_node_map_.emplace(config.client_id, client_node_ptr);
  }

  ClientFrameNode::Ptr getClientNodePtrById(const CliId& cid) const {
    auto it = client_frame_node_map_.find(cid);
    if (it != client_frame_node_map_.end()) {
      return it->second;
    } else {
      return nullptr;
    }
  }

  ceres::LocalParameterization* getLocalParameterization() const {
    return local_parameterization_.get();
  }

  const ClientFrameNodeMap& getClientNodes() { return client_frame_node_map_; }

 private:
  using Node = voxgraph::Node;

  Node::NodeId node_id_counter_ = 0;
  const Node::NodeId newNodeId() { return node_id_counter_++; }

  ClientFrameNodeMap client_frame_node_map_;

  std::shared_ptr<ceres::LocalParameterization> local_parameterization_;
};

}  // namespace server

}  // namespace coxgraph

#endif  // COXGRAPH_SERVER_BACKEND_NODE_COLLECTION_H_
