#ifndef COXGRAPH_SERVER_BACKEND_CLIENT_FRAME_NODE_H_
#define COXGRAPH_SERVER_BACKEND_CLIENT_FRAME_NODE_H_

#include <voxgraph/backend/node/node.h>

#include <memory>

#include "coxgraph/common.h"

namespace coxgraph {
namespace server {

class ClientFrameNode : public voxgraph::Node {
 public:
  typedef std::shared_ptr<ClientFrameNode> Ptr;

  struct Config : Node::Config {
    CliId client_id;
  };

  ClientFrameNode(const NodeId& node_id, const Config& config)
      : Node(node_id, config), config_(config) {}
  ~ClientFrameNode() = default;

  CliId getCliId() const { return config_.client_id; }

 private:
  using Node = voxgraph::Node;

  Config config_;
};

}  // namespace server
}  // namespace coxgraph
#endif  // COXGRAPH_SERVER_BACKEND_CLIENT_FRAME_NODE_H_
