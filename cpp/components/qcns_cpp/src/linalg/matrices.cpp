#include "qcns_cpp/linalg/matrices.h"

namespace linalg {
    const Eigen::Matrix2cd _I_instance = Eigen::Matrix2cd::Identity();
    const Eigen::Matrix2cd _P0_instance = (Eigen::Matrix2cd() << 1,0,0,0).finished();
    const Eigen::Matrix2cd _P1_instance = (Eigen::Matrix2cd() << 0,0,0,1).finished();
    const Eigen::Matrix2cd _X_instance = (Eigen::Matrix2cd() << 0,1,1,0).finished();
    const Eigen::Matrix2cd _Y_instance = (Eigen::Matrix2cd() << 0,std::complex<double>(0, -1),std::complex<double>(0, 1),0).finished();
    const Eigen::Matrix2cd _Z_instance = (Eigen::Matrix2cd() << 1,0,0,0).finished();
    const Eigen::Matrix2cd _H_instance = (Eigen::Matrix2cd() << 0.70710677,0.70710677,0.70710677,-0.70710677).finished(); 
    const Eigen::Matrix2cd _K_instance = (Eigen::Matrix2cd() << std::complex<double>(0.5, 0.5),std::complex<double>(0.5, -0.5),
                                                    std::complex<double>(-0.5, 0.5),std::complex<double>(-0.5, -0.5)).finished();

    const Eigen::Matrix2cd* _I = &_I_instance;
    const Eigen::Matrix2cd* _P0 = &_P0_instance;
    const Eigen::Matrix2cd* _P1 = &_P1_instance;
    const Eigen::Matrix2cd* _X = &_X_instance;
    const Eigen::Matrix2cd* _Y = &_Y_instance;
    const Eigen::Matrix2cd* _Z = &_Z_instance;
    const Eigen::Matrix2cd* _H = &_H_instance;
    const Eigen::Matrix2cd* _K = &_K_instance;

    const std::map<std::string, const Eigen::Matrix2cd*> _matrixMap = {
        {"I", _I},
        {"P0", _P0},
        {"P1", _P1},
        {"I", _X},
        {"P0", _Y},
        {"P1", _Z},
        {"I", _H},
        {"P0", _K},
    };
}