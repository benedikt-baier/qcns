#include "operator_utils.h"
#include "matrix_utils.h"
#include "matrices.h"


namespace linear_algebra {

// get_single_operator
Eigen::MatrixXcd 
get_single_operator(const bool& sparse,  const Eigen::Matrix2cd& gate, const uint32_t& index, const uint32_t& num_qubits)
{
    /*
    Calculates the operator for a gate applied to a single qubit

    Args:
        key (string): key for caching the operator
        gate (matrix_c_d): gate to apply to the qubit
        index (uint32_t): index of the qubit
        num_qubits (uint32_t): number of qubits in the system

    Returns:
        gate_f (matrix_c_d): resulting operator
    */

    if (num_qubits == 1)
        return gate;

    std::vector<MatrixTypeVariant> op(num_qubits, I);
    op[index] = gate;
    Eigen::MatrixXcd gate_f = tensor_operator(op);

    return gate_f;
}

Eigen::MatrixXd 
get_single_operator(const bool& sparse,  const Eigen::Matrix2d& gate, const uint32_t& index, const uint32_t& num_qubits)
{
    /*
    Calculates the operator for a gate applied to a single qubit

    Args:
        key (string): key for caching the operator
        gate (matrix_c_d): gate to apply to the qubit
        index (uint32_t): index of the qubit
        num_qubits (uint32_t): number of qubits in the system

    Returns:
        gate_f (matrix_c_d): resulting operator
    */

    if (num_qubits == 1)
        return gate;

    std::vector<Eigen::Matrix2d> op(num_qubits, I);
    op[index] = gate;
    Eigen::MatrixXd gate_f = tensor_operator(op);

    return gate_f;
}

// tensor_operator
Eigen::MatrixXcd
tensor_operator(const std::vector<MatrixTypeVariant>& op)
{
    /*
    Performs the kronecker product of all the matrices in the given vector

    Args:
        op (vector<matrix_c_d>): vector of matrices

    Returns:
        res (matrix_c_d): result of kronecker products
    */
    Eigen::MatrixXcd res = kron(op[0], op[1]);
    for (uint32_t i = 2; i < op.size(); i++)
        res = kron(res, op[i]);
    // TODO idea to speed up
    return res;
}

Eigen::MatrixXd
tensor_operator(const std::vector<Eigen::Matrix2d>& op)
{
    /*
    Performs the kronecker product of all the matrices in the given vector

    Args:
        op (vector<matrix_c_d>): vector of matrices

    Returns:
        res (matrix_c_d): result of kronecker products
    */
    Eigen::MatrixXd res = kron(op[0], op[1]);
    for (uint32_t i = 2; i < op.size(); i++)
        res = kron(res, op[i]);
    // TODO idea to speed up
    return res;
}
}