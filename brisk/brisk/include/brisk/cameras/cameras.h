/*
 * cameras.hpp
 *
 *  Created on: Dec 25, 2013
 *      Author: lestefan
 */

#ifndef CAMERAS_HPP_
#define CAMERAS_HPP_

#include <brisk/cameras/pinhole-camera-geometry.h>
#include <brisk/cameras/no-distortion.h>
#include <brisk/cameras/radial-tangential-distortion.h>
#include <brisk/cameras/equidistant-distortion.h>

namespace brisk {
namespace cameras {

// some main camera geometries
typedef PinholeCameraGeometry<NoDistortion> UndistortedPinholeCameraGeometry;
typedef PinholeCameraGeometry<RadialTangentialDistortion> RadialTangentialPinholeCameraGeometry;
typedef PinholeCameraGeometry<EquidistantDistortion> EquidistantPinholeCameraGeometry;

}  // namespace cameras
}  // namespace brisk

#endif /* CAMERAS_HPP_ */
