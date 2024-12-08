#pragma once


#include <Eigen/Dense>
#include <map>
#include <string>

namespace linalg {
    extern const Eigen::Matrix2cd* _I;
    extern const Eigen::Matrix2cd* _P0;
    extern const Eigen::Matrix2cd* _P1;
    extern const Eigen::Matrix2cd* _X;
    extern const Eigen::Matrix2cd* _Y;
    extern const Eigen::Matrix2cd* _Z;
    extern const Eigen::Matrix2cd* _H;
    extern const Eigen::Matrix2cd* _K;
    extern const std::map<std::string, const Eigen::Matrix2cd*> _matrixMap;

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
