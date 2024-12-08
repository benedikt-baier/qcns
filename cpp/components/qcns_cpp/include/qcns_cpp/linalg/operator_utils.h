#pragma once

#include <Eigen/Dense>

namespace linalg {

    // Single Operator
    Eigen::MatrixXcd // Complex Operator
    get_single_operator(const std::string& key, const Eigen::Matrix2cd& gate, const uint32_t& index, const uint32_t& num_qubits);

    Eigen::MatrixXd  // Real Operator
    get_single_operator(const std::string& key, const Eigen::Matrix2d& gate, const uint32_t& index, const uint32_t& num_qubits);

    // Tensor Operator
    Eigen::MatrixXcd // Complex Operator
    tensor_operator(const Eigen::Vector2d& idendity_cnt, const Eigen::Matrix2cd& op);

    Eigen::MatrixXd // Real Operator
    tensor_operator(const Eigen::Vector2d& idendity_cnt, const Eigen::Matrix2d& op);

    Eigen::MatrixXcd tensor_operator(const std::vector<std::string>& keys);

}