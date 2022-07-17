#ifndef COXGRAPH_UTILS_ROS_PARAMS_H_
#define COXGRAPH_UTILS_ROS_PARAMS_H_

#include "coxgraph/common.h"

namespace coxgraph {
namespace utils {

inline void setInformationMatrixFromRosParams(
    const ros::NodeHandle& node_handle, InformationMatrix* information_matrix) {
  CHECK_NOTNULL(information_matrix);
  InformationMatrix& information_matrix_ref = *information_matrix;

  // Set the upper triangular part of the information matrix from ROS params
  node_handle.param("x_x", information_matrix_ref(0, 0), 0.0);
  node_handle.param("x_y", information_matrix_ref(0, 1), 0.0);
  node_handle.param("x_z", information_matrix_ref(0, 2), 0.0);
  node_handle.param("x_yaw", information_matrix_ref(0, 3), 0.0);

  node_handle.param("y_y", information_matrix_ref(1, 1), 0.0);
  node_handle.param("y_z", information_matrix_ref(1, 2), 0.0);
  node_handle.param("y_yaw", information_matrix_ref(1, 3), 0.0);

  node_handle.param("z_z", information_matrix_ref(2, 2), 0.0);
  node_handle.param("z_yaw", information_matrix_ref(2, 3), 0.0);

  node_handle.param("yaw_yaw", information_matrix_ref(3, 3), 0.0);

  // Copy the upper to the lower triangular part, to get a symmetric info matrix
  information_matrix_ref =
      information_matrix_ref.selfadjointView<Eigen::Upper>();
}
}  // namespace utils
}  // namespace coxgraph

#endif  // COXGRAPH_UTILS_ROS_PARAMS_H_
