#ifndef COXGRAPH_MOD_COMMON_H_
#define COXGRAPH_MOD_COMMON_H_

#include <Eigen/Dense>

#include <functional>

namespace coxgraph {
namespace mod {

typedef std::function<void(Eigen::Matrix4d, double)> TfPubFunc;
typedef std::function<void(size_t, size_t, double, double, Eigen::Matrix4d)>
    MfPubFunc;
typedef std::function<void(double, double, Eigen::Matrix4d)> LcPubFunc;

typedef int8_t CliId;
}  // namespace mod
}  // namespace coxgraph

#endif  // COXGRAPH_MOD_COMMON_H_
