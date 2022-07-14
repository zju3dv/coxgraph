/*
 * RadialTangentialDistortion.hpp
 *
 *  Created on: Dec 25, 2013
 *      Author: lestefan
 */

namespace brisk {
namespace cameras {

RadialTangentialDistortion::RadialTangentialDistortion(double k1, double k2,
                                                       double p1, double p2)
    : _k1(k1),
      _k2(k2),
      _p1(p1),
      _p2(p2) {
}

// distort an undistorted point
void RadialTangentialDistortion::distort(Point2d& point) const {
  double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

  mx2_u = point[0] * point[0];
  my2_u = point[1] * point[1];
  mxy_u = point[0] * point[1];
  rho2_u = mx2_u + my2_u;
  rad_dist_u = _k1 * rho2_u + _k2 * rho2_u * rho2_u;
  point[0] += point[0] * rad_dist_u + 2.0 * _p1 * mxy_u
      + _p2 * (rho2_u + 2.0 * mx2_u);
  point[1] += point[1] * rad_dist_u + 2.0 * _p2 * mxy_u
      + _p1 * (rho2_u + 2.0 * my2_u);
}

void RadialTangentialDistortion::distort(Point2d& point,
                                         Matx22d& jacobian_out) const {
  double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

  mx2_u = point[0] * point[0];
  my2_u = point[1] * point[1];
  mxy_u = point[0] * point[1];
  rho2_u = mx2_u + my2_u;

  rad_dist_u = _k1 * rho2_u + _k2 * rho2_u * rho2_u;

  jacobian_out(0, 0) = 1 + rad_dist_u + _k1 * 2.0 * mx2_u
      + _k2 * rho2_u * 4 * mx2_u + 2.0 * _p1 * point[1] + 6 * _p2 * point[0];
  jacobian_out(1, 0) = _k1 * 2.0 * point[0] * point[1]
      + _k2 * 4 * rho2_u * point[0] * point[1] + _p1 * 2.0 * point[0]
      + 2.0 * _p2 * point[1];
  jacobian_out(0, 1) = jacobian_out(1, 0);
  jacobian_out(1, 1) = 1 + rad_dist_u + _k1 * 2.0 * my2_u
      + _k2 * rho2_u * 4 * my2_u + 6 * _p1 * point[1] + 2.0 * _p2 * point[0];

  point[0] += point[0] * rad_dist_u + 2.0 * _p1 * mxy_u
      + _p2 * (rho2_u + 2.0 * mx2_u);
  point[1] += point[1] * rad_dist_u + 2.0 * _p2 * mxy_u
      + _p1 * (rho2_u + 2.0 * my2_u);

}

// undistort a distorted point
void RadialTangentialDistortion::undistort(Point2d& point) const {
  Point2d ybar = point;
  const int n = 5;
  Matx22d F;

  Point2d y_tmp;

  for (int i = 0; i < n; i++) {

    y_tmp = ybar;

    distort(y_tmp, F);

    Point2d e(point - y_tmp);
    Point2d du = (F.t() * F).inv() * F.t() * e;

    ybar += du;

    if (e.dot(e) < 1e-15)
      break;

  }
  point = ybar;
}

void RadialTangentialDistortion::undistort(
    Point2d& point, Matx22d& inverse_jacobian_out) const {
  // we use f^-1 ' = ( f'(f^-1) ) '
  // with f^-1 the undistortion
  // and  f the distortion
  undistort(point);  // first get the undistorted image

  Point2d kp = point;
  Matx22d Jd;
  distort(kp, Jd);

  // now y = f^-1(y0)
  inverse_jacobian_out = Jd.inv();
}

}
}
