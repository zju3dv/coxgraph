/*
 * EquidistantDistortion.hpp
 *
 *  Created on: Dec 26, 2013
 *      Author: lestefan
 */

namespace brisk {
namespace cameras {

EquidistantDistortion::EquidistantDistortion(double k1, double k2, double k3,
                                             double k4)
    : _k1(k1),
      _k2(k2),
      _k3(k3),
      _k4(k4) {
}

void EquidistantDistortion::distort(Point2d& point) const {

  double r, theta, theta2, theta4, theta6, theta8, thetad, scaling;

  r = sqrt(point[0] * point[0] + point[1] * point[1]);
  theta = atan(r);
  theta2 = theta * theta;
  theta4 = theta2 * theta2;
  theta6 = theta4 * theta2;
  theta8 = theta4 * theta4;
  thetad = theta
      * (1 + _k1 * theta2 + _k2 * theta4 + _k3 * theta6 + _k4 * theta8);

  scaling = (r > 1e-8) ? thetad / r : 1.0;
  point[0] *= scaling;
  point[1] *= scaling;
}

void EquidistantDistortion::distort(Point2d& point,
                                    Matx22d& jacobian_out) const {

  double r, theta, theta2, theta4, theta6, theta8, thetad, scaling;

  //MATLAB generated Jacobian
  jacobian_out(0, 0) = atan(sqrt(point[0] * point[0] + point[1] * point[1]))
      * 1.0 / sqrt(point[0] * point[0] + point[1] * point[1])
      * (_k1 * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])), 2.0)
          + _k2
              * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])), 4.0)
          + _k3
              * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])), 6.0)
          + _k4
              * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])), 8.0)
          + 1.0)
      + point[0] * atan(sqrt(point[0] * point[0] + point[1] * point[1])) * 1.0
          / sqrt(point[0] * point[0] + point[1] * point[1])
          * ((_k2 * point[0]
              * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])), 3.0)
              * 1.0 / sqrt(point[0] * point[0] + point[1] * point[1]) * 4.0)
              / (point[0] * point[0] + point[1] * point[1] + 1.0)
              + (_k3 * point[0]
                  * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])),
                        5.0) * 1.0
                  / sqrt(point[0] * point[0] + point[1] * point[1]) * 6.0)
                  / (point[0] * point[0] + point[1] * point[1] + 1.0)
              + (_k4 * point[0]
                  * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])),
                        7.0) * 1.0
                  / sqrt(point[0] * point[0] + point[1] * point[1]) * 8.0)
                  / (point[0] * point[0] + point[1] * point[1] + 1.0)
              + (_k1 * point[0]
                  * atan(sqrt(point[0] * point[0] + point[1] * point[1])) * 1.0
                  / sqrt(point[0] * point[0] + point[1] * point[1]) * 2.0)
                  / (point[0] * point[0] + point[1] * point[1] + 1.0))
      + ((point[0] * point[0])
          * (_k1
              * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])), 2.0)
              + _k2
                  * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])),
                        4.0)
              + _k3
                  * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])),
                        6.0)
              + _k4
                  * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])),
                        8.0) + 1.0))
          / ((point[0] * point[0] + point[1] * point[1])
              * (point[0] * point[0] + point[1] * point[1] + 1.0))
      - (point[0] * point[0])
          * atan(sqrt(point[0] * point[0] + point[1] * point[1])) * 1.0
          / pow(point[0] * point[0] + point[1] * point[1], 3.0 / 2.0)
          * (_k1
              * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])), 2.0)
              + _k2
                  * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])),
                        4.0)
              + _k3
                  * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])),
                        6.0)
              + _k4
                  * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])),
                        8.0) + 1.0);
  jacobian_out(0, 1) = point[0]
      * atan(sqrt(point[0] * point[0] + point[1] * point[1])) * 1.0
      / sqrt(point[0] * point[0] + point[1] * point[1])
      * ((_k2 * point[1]
          * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])), 3.0)
          * 1.0 / sqrt(point[0] * point[0] + point[1] * point[1]) * 4.0)
          / (point[0] * point[0] + point[1] * point[1] + 1.0)
          + (_k3 * point[1]
              * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])), 5.0)
              * 1.0 / sqrt(point[0] * point[0] + point[1] * point[1]) * 6.0)
              / (point[0] * point[0] + point[1] * point[1] + 1.0)
          + (_k4 * point[1]
              * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])), 7.0)
              * 1.0 / sqrt(point[0] * point[0] + point[1] * point[1]) * 8.0)
              / (point[0] * point[0] + point[1] * point[1] + 1.0)
          + (_k1 * point[1]
              * atan(sqrt(point[0] * point[0] + point[1] * point[1])) * 1.0
              / sqrt(point[0] * point[0] + point[1] * point[1]) * 2.0)
              / (point[0] * point[0] + point[1] * point[1] + 1.0))
      + (point[0] * point[1]
          * (_k1
              * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])), 2.0)
              + _k2
                  * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])),
                        4.0)
              + _k3
                  * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])),
                        6.0)
              + _k4
                  * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])),
                        8.0) + 1.0))
          / ((point[0] * point[0] + point[1] * point[1])
              * (point[0] * point[0] + point[1] * point[1] + 1.0))
      - point[0] * point[1]
          * atan(sqrt(point[0] * point[0] + point[1] * point[1])) * 1.0
          / pow(point[0] * point[0] + point[1] * point[1], 3.0 / 2.0)
          * (_k1
              * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])), 2.0)
              + _k2
                  * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])),
                        4.0)
              + _k3
                  * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])),
                        6.0)
              + _k4
                  * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])),
                        8.0) + 1.0);
  jacobian_out(1, 0) = point[1]
      * atan(sqrt(point[0] * point[0] + point[1] * point[1])) * 1.0
      / sqrt(point[0] * point[0] + point[1] * point[1])
      * ((_k2 * point[0]
          * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])), 3.0)
          * 1.0 / sqrt(point[0] * point[0] + point[1] * point[1]) * 4.0)
          / (point[0] * point[0] + point[1] * point[1] + 1.0)
          + (_k3 * point[0]
              * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])), 5.0)
              * 1.0 / sqrt(point[0] * point[0] + point[1] * point[1]) * 6.0)
              / (point[0] * point[0] + point[1] * point[1] + 1.0)
          + (_k4 * point[0]
              * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])), 7.0)
              * 1.0 / sqrt(point[0] * point[0] + point[1] * point[1]) * 8.0)
              / (point[0] * point[0] + point[1] * point[1] + 1.0)
          + (_k1 * point[0]
              * atan(sqrt(point[0] * point[0] + point[1] * point[1])) * 1.0
              / sqrt(point[0] * point[0] + point[1] * point[1]) * 2.0)
              / (point[0] * point[0] + point[1] * point[1] + 1.0))
      + (point[0] * point[1]
          * (_k1
              * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])), 2.0)
              + _k2
                  * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])),
                        4.0)
              + _k3
                  * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])),
                        6.0)
              + _k4
                  * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])),
                        8.0) + 1.0))
          / ((point[0] * point[0] + point[1] * point[1])
              * (point[0] * point[0] + point[1] * point[1] + 1.0))
      - point[0] * point[1]
          * atan(sqrt(point[0] * point[0] + point[1] * point[1])) * 1.0
          / pow(point[0] * point[0] + point[1] * point[1], 3.0 / 2.0)
          * (_k1
              * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])), 2.0)
              + _k2
                  * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])),
                        4.0)
              + _k3
                  * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])),
                        6.0)
              + _k4
                  * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])),
                        8.0) + 1.0);
  jacobian_out(1, 1) = atan(sqrt(point[0] * point[0] + point[1] * point[1]))
      * 1.0 / sqrt(point[0] * point[0] + point[1] * point[1])
      * (_k1 * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])), 2.0)
          + _k2
              * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])), 4.0)
          + _k3
              * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])), 6.0)
          + _k4
              * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])), 8.0)
          + 1.0)
      + point[1] * atan(sqrt(point[0] * point[0] + point[1] * point[1])) * 1.0
          / sqrt(point[0] * point[0] + point[1] * point[1])
          * ((_k2 * point[1]
              * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])), 3.0)
              * 1.0 / sqrt(point[0] * point[0] + point[1] * point[1]) * 4.0)
              / (point[0] * point[0] + point[1] * point[1] + 1.0)
              + (_k3 * point[1]
                  * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])),
                        5.0) * 1.0
                  / sqrt(point[0] * point[0] + point[1] * point[1]) * 6.0)
                  / (point[0] * point[0] + point[1] * point[1] + 1.0)
              + (_k4 * point[1]
                  * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])),
                        7.0) * 1.0
                  / sqrt(point[0] * point[0] + point[1] * point[1]) * 8.0)
                  / (point[0] * point[0] + point[1] * point[1] + 1.0)
              + (_k1 * point[1]
                  * atan(sqrt(point[0] * point[0] + point[1] * point[1])) * 1.0
                  / sqrt(point[0] * point[0] + point[1] * point[1]) * 2.0)
                  / (point[0] * point[0] + point[1] * point[1] + 1.0))
      + ((point[1] * point[1])
          * (_k1
              * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])), 2.0)
              + _k2
                  * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])),
                        4.0)
              + _k3
                  * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])),
                        6.0)
              + _k4
                  * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])),
                        8.0) + 1.0))
          / ((point[0] * point[0] + point[1] * point[1])
              * (point[0] * point[0] + point[1] * point[1] + 1.0))
      - (point[1] * point[1])
          * atan(sqrt(point[0] * point[0] + point[1] * point[1])) * 1.0
          / pow(point[0] * point[0] + point[1] * point[1], 3.0 / 2.0)
          * (_k1
              * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])), 2.0)
              + _k2
                  * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])),
                        4.0)
              + _k3
                  * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])),
                        6.0)
              + _k4
                  * pow(atan(sqrt(point[0] * point[0] + point[1] * point[1])),
                        8.0) + 1.0);

  r = sqrt(point[0] * point[0] + point[1] * point[1]);
  theta = atan(r);
  theta2 = theta * theta;
  theta4 = theta2 * theta2;
  theta6 = theta4 * theta2;
  theta8 = theta4 * theta4;
  thetad = theta
      * (1 + _k1 * theta2 + _k2 * theta4 + _k3 * theta6 + _k4 * theta8);

  scaling = (r > 1e-8) ? thetad / r : 1.0;
  point[0] *= scaling;
  point[1] *= scaling;
}

void EquidistantDistortion::undistort(Point2d& point) const {

  double theta, theta2, theta4, theta6, theta8, thetad, scaling;

  thetad = sqrt(point[0] * point[0] + point[1] * point[1]);
  theta = thetad;  // initial guess
  for (int i = 20; i > 0; i--) {
    theta2 = theta * theta;
    theta4 = theta2 * theta2;
    theta6 = theta4 * theta2;
    theta8 = theta4 * theta4;
    theta = thetad
        / (1 + _k1 * theta2 + _k2 * theta4 + _k3 * theta6 + _k4 * theta8);
  }
  scaling = tan(theta) / thetad;

  point[0] *= scaling;
  point[1] *= scaling;
}

void EquidistantDistortion::undistort(Point2d& point,
                                      Matx22d& jacobian_out) const {

  // we use f^-1 ' = ( f'(f^-1) ) '
  // with f^-1 the undistortion
  // and  f the distortion
  undistort(point);  // first get the undistorted image

  Point2d kp = point;  // copy...
  Matx22d Jd;
  distort(kp, Jd);

  // now y = f^-1(y0)

  jacobian_out = Jd.inv();

}

}
}
