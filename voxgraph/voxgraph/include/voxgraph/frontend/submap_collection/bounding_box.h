#ifndef VOXGRAPH_FRONTEND_SUBMAP_COLLECTION_BOUNDING_BOX_H_
#define VOXGRAPH_FRONTEND_SUBMAP_COLLECTION_BOUNDING_BOX_H_

#include <voxblox/core/common.h>
#include <Eigen/Dense>

#include <algorithm>

namespace voxgraph {
typedef Eigen::Matrix<voxblox::FloatingPoint, 3, 8> BoxCornerMatrix;
class BoundingBox {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  voxblox::Point min = {INFINITY, INFINITY, INFINITY};
  voxblox::Point max = {-INFINITY, -INFINITY, -INFINITY};

  void reset();

  const BoxCornerMatrix getCornerCoordinates() const;

  static const BoundingBox getAabbFromObbAndPose(
      const BoundingBox& obb, const voxblox::Transformation& pose);

  bool overlapsWith(const BoundingBox& other_bounding_box) const {
    // If there's a separation along any of the 3 axes, the AABBs don't
    // intersect
    if (max[0] < other_bounding_box.min[0] ||
        min[0] > other_bounding_box.max[0])
      return false;
    if (max[1] < other_bounding_box.min[1] ||
        min[1] > other_bounding_box.max[1])
      return false;
    if (max[2] < other_bounding_box.min[2] ||
        min[2] > other_bounding_box.max[2])
      return false;
    // Since the AABBs overlap on all axes, the submaps could be overlapping
    return true;
  }

  /**
   * @brief Compute overlapping ratio between this and other submap
   *
   * @param other_bounding_box
   * @return float overlapping ratio
   */
  float overlapRatioWith(const BoundingBox& other_bounding_box) const {
    float overlap_xyz[3] = {0, 0, 0};
    if (max[0] > other_bounding_box.min[0] &&
        min[0] < other_bounding_box.max[0])
      overlap_xyz[0] = std::min(std::fabs(max[0] - other_bounding_box.min[0]),
                                std::fabs(min[0] - other_bounding_box.max[0]));
    if (max[1] > other_bounding_box.min[1] &&
        min[1] < other_bounding_box.max[1])
      overlap_xyz[1] = std::min(std::fabs(max[1] - other_bounding_box.min[1]),
                                std::fabs(min[1] - other_bounding_box.max[1]));
    if (max[2] > other_bounding_box.min[2] &&
        min[2] < other_bounding_box.max[2])
      overlap_xyz[2] = std::min(std::fabs(max[2] - other_bounding_box.min[2]),
                                std::fabs(min[2] - other_bounding_box.max[2]));

    float overlap_vol = overlap_xyz[0] * overlap_xyz[1] * overlap_xyz[2];
    float curr_vol = (max[0] - min[0]) * (max[1] - min[1]) * (max[2] - max[2]);

    if (overlap_vol > 0)
      return overlap_vol / curr_vol;
    else
      return 0;
  }

  bool hasPosition(voxblox::Point pos) {
    if (pos[0] < max[0] && pos[0] > min[0] && pos[1] < max[1] &&
        pos[1] > min[1] && pos[2] < max[2] && pos[2] > min[2])
      return true;
    else
      return false;
  }

  float distToPosition(voxblox::Point pos) {
    return ((max + min) / 2 - pos).norm();
  }
};
}  // namespace voxgraph

#endif  // VOXGRAPH_FRONTEND_SUBMAP_COLLECTION_BOUNDING_BOX_H_
