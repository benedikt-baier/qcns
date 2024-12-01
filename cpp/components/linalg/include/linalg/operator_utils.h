#ifndef OPERATOR_UTILS_H
#define OPERATOR_UTILS_H

#include <Eigen/Dense>
#include <vector>

namespace linear_algebra {

Eigen::MatrixXcd 
get_single_operator(const bool& sparse, const uint32_t& index, const uint32_t& num_qubits);

Eigen::MatrixXcd
tensor_operator(const std::vector<Eigen::MatrixXd>& op);

}

#endif