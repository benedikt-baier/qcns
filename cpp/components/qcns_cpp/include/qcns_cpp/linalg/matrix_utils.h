#pragma once

#include <Eigen/Dense>

namespace linalg {

/*
Add Functions
*/
Eigen::MatrixXcd 
add(const Eigen::MatrixXcd& mat1, const Eigen::MatrixXcd& mat2);

Eigen::MatrixXd 
add(const Eigen::MatrixXd& mat1, const Eigen::MatrixXd& mat2);

Eigen::MatrixXcd 
add(const Eigen::MatrixXcd& mat1, const Eigen::MatrixXd& mat2);

Eigen::MatrixXcd 
add(const Eigen::MatrixXd& mat1, const Eigen::MatrixXcd& mat2);

/*
Multiply Functions
*/
void multi_dot(const Eigen::MatrixXcd& transform_mat, const Eigen::MatrixXcd& target_mat);
void multi_dot(const Eigen::MatrixXd& transform_mat, const Eigen::MatrixXd& target_mat);
// Eigen::MatrixXd 
// multiply(const Eigen::MatrixXd& mat1, const Eigen::MatrixXd& mat2);

/*
Kronecker Product Functions
*/

template<typename A, typename B>
Eigen::MatrixXcd kron(const Eigen::MatrixBase<A>& mat1, const Eigen::MatrixBase<B>& mat2);
// Grenzfall I*I*I*G_2x2 complex
Eigen::MatrixXcd kron_idendity_left(const Eigen::MatrixXcd& I, const Eigen::Matrix2cd& mat);   

// Grenzfall I*I*I*G_2x2 real
Eigen::MatrixXd kron_idendity_left(const Eigen::MatrixXd& I, const Eigen::Matrix2d& mat);   

// Grenzfall I*I*I*G_NxN complex
Eigen::MatrixXcd kron_idendity_left(const Eigen::MatrixXcd& I, const Eigen::MatrixXcd& mat);   

// Grenzfall I*I*I*G_NxN real
Eigen::MatrixXd kron_idendity_left(const Eigen::MatrixXd& I, const Eigen::MatrixXd& mat);   

// Grenzfall: G_2x2 * I * I * I complex
Eigen::MatrixXcd kron_idendity_right(const Eigen::Matrix2cd& mat, const Eigen::MatrixXcd& I);

// Grenzfall: G_2x2 * I * I * I real
Eigen::MatrixXd kron_idendity_right(const Eigen::Matrix2d& mat, const Eigen::MatrixXd& I); 

// Grenzfall: G_NxN * I * I * I complex
Eigen::MatrixXcd kron_idendity_right(const Eigen::MatrixXcd& mat, const Eigen::MatrixXcd& I);

// Grenzfall: G_NxN * I * I * I real
Eigen::MatrixXd kron_idendity_right(const Eigen::MatrixXd& mat, const Eigen::MatrixXd& I); 

// Eigen::MatrixXd
// initializeRightKronMatrix(const uint32_t& rows, const uint32_t& cols);

// template<typename MatrixType1, typename MatrixType2>
// Eigen::MatrixXcd kron(const Eigen::MatrixBase<MatrixType1>& mat1, const Eigen::MatrixBase<MatrixType2>& mat2);

}