#include "matrix_utils.h"
#include <unsupported/Eigen/KroneckerProduct>

namespace linear_algebra {

// Add
Eigen::MatrixXcd 
add(const Eigen::MatrixXcd& mat1, const Eigen::MatrixXcd& mat2) 
{
    return mat1 + mat2;
}
Eigen::MatrixXd 
add(const Eigen::MatrixXd& mat1, const Eigen::MatrixXd& mat2) 
{
    return mat1 + mat2;
}
Eigen::MatrixXcd 
add(const Eigen::MatrixXcd& mat1, const Eigen::MatrixXd& mat2) 
{
    return mat1 + mat2.cast<std::complex<double>>();
}
Eigen::MatrixXcd 
add(const Eigen::MatrixXd& mat1, const Eigen::MatrixXcd& mat2) 
{
    return mat1.cast<std::complex<double>>() + mat2;
}

// Multiply
Eigen::MatrixXd 
multiply(const Eigen::MatrixXd& mat1, const Eigen::MatrixXd& mat2) 
{
    return mat1 * mat2;
}

// Kron
Eigen::MatrixXcd kron(const Eigen::MatrixXcd& mat1, const Eigen::MatrixXcd& mat2)
{
    return Eigen::kroneckerProduct(mat1, mat2).eval();
}
Eigen::MatrixXd kron(const Eigen::MatrixXd& mat1, const Eigen::MatrixXd& mat2)
{
    return Eigen::kroneckerProduct(mat1, mat2).eval();
}
Eigen::MatrixXcd kron(const Eigen::MatrixXcd& mat1, const Eigen::MatrixXd& mat2)
{
    return Eigen::kroneckerProduct(mat1, mat2.cast<std::complex<double>>()).eval();
}
Eigen::MatrixXcd kron(const Eigen::MatrixXd& mat1, const Eigen::MatrixXcd& mat2)
{
    return Eigen::kroneckerProduct(mat1.cast<std::complex<double>>(), mat2).eval();
}


}