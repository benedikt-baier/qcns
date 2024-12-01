#ifndef MATRICES_H
#define MATRICES_H

#include <Eigen/Dense>

namespace linear_algebra {

const Eigen::Matrix2d I = Eigen::Matrix2d::Identity();
const Eigen::Matrix2d P0 = (Eigen::Matrix2d() << 1,0,0,0).finished();
const Eigen::Matrix2d P1 = (Eigen::Matrix2d() << 0,0,0,1).finished();
const Eigen::Matrix2d X = (Eigen::Matrix2d() << 0,1,1,0).finished();
const Eigen::Matrix2cd Y = (Eigen::Matrix2cd() << 0,std::complex<double>(0, -1),std::complex<double>(0, 1),0).finished();
const Eigen::Matrix2d Z = (Eigen::Matrix2d() << 1,0,0,0).finished();
const Eigen::Matrix2d H = (Eigen::Matrix2d() << 0.70710677,0.70710677,0.70710677,-0.70710677).finished(); 
const Eigen::Matrix2cd K = (Eigen::Matrix2cd() << std::complex<double>(0.5, 0.5),std::complex<double>(0.5, -0.5),
                                                  std::complex<double>(-0.5, 0.5),std::complex<double>(-0.5, -0.5)).finished();
// const Eigen::Matrix2cd SX = ;
// const Eigen::Matrix2cd SY = ;
// const Eigen::Matrix2cd SZ = ;
// const Eigen::Matrix2cd T = ;
// const Eigen::Matrix2cd iSX = ;
// const Eigen::Matrix2cd iSY = ;
// const Eigen::Matrix2cd iSZ = ;
// const Eigen::Matrix2cd iT = ;
// const Eigen::Matrix2cd iK = ;
}

#endif