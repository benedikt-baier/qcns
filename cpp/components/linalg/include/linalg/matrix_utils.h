#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <Eigen/Dense>
#include <variant>

namespace linear_algebra {

// Matrix Type
using MatrixTypeVariant = std::variant<Eigen::Matrix2d, Eigen::Matrix2cd>;
// Add
Eigen::MatrixXcd 
add(const Eigen::MatrixXcd& mat1, const Eigen::MatrixXcd& mat2);
Eigen::MatrixXd 
add(const Eigen::MatrixXd& mat1, const Eigen::MatrixXd& mat2);
Eigen::MatrixXcd 
add(const Eigen::MatrixXcd& mat1, const Eigen::MatrixXd& mat2);
Eigen::MatrixXcd 
add(const Eigen::MatrixXd& mat1, const Eigen::MatrixXcd& mat2);

// Multiply
Eigen::MatrixXd 
multiply(const Eigen::MatrixXd& mat1, const Eigen::MatrixXd& mat2);

// Kron
Eigen::MatrixXd 
kron(const Eigen::MatrixXd& mat1, const Eigen::MatrixXd& mat2);
    
}

#endif