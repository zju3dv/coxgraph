#include "coxgraph/server/client_tf_optimizer.h"

#include "coxgraph/common.h"

namespace coxgraph {
namespace server {

void ClientTfOptimizer::addClient(const CliId& cid,
                                  const Transformation& pose) {
  CHECK(!pose_graph_.hasClientNode(cid));

  ClientFrameNode::Config config;
  config.client_id = cid;
  if (cid == 0)
    config.set_constant = true;
  else
    config.set_constant = false;
  config.T_I_node_initial = pose;

  pose_graph_.addClientNode(config);
}

void ClientTfOptimizer::addClientRelativePoseMeasurement(
    const CliId& first_cid, const CliId& second_cid,
    const Transformation& T_C1_C2) {
  RelativePoseConstraint::Config config;
  config.information_matrix = cli_rp_info_matrix_;
  config.origin_client_id = first_cid;
  config.destination_client_id = second_cid;
  config.T_origin_destination = T_C1_C2;

  pose_graph_.addClientRelativePoseConstraint(config);
}

}  // namespace server
}  // namespace coxgraph
