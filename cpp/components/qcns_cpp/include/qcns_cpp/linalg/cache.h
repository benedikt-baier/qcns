#pragma once
#include <unordered_map>
#include <iostream>
#include <Eigen/Dense>

namespace linalg {
    std::unordered_map<std::string, Eigen::MatrixXcd> s_complex_cache;
    std::unordered_map<std::string, Eigen::MatrixXd> s_real_cache;
    std::unordered_map<std::string, Eigen::MatrixXcd> d_complex_cache;
    std::unordered_map<std::string, Eigen::MatrixXd> d_real_cache;
    std::unordered_map<std::string, Eigen::MatrixXcd> t_complex_cache;
    std::unordered_map<std::string, Eigen::MatrixXd> t_real_cache;
}