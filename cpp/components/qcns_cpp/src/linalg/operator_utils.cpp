#include "qcns_cpp/linalg/operator_utils.h"
#include "qcns_cpp/linalg/matrices.h"
#include "qcns_cpp/linalg/matrix_utils.h"
#include "qcns_cpp/linalg/cache.h"

namespace linalg {

    /*
    get_single_operator functions
    */

    Eigen::MatrixXcd // Complex Operator
    get_single_operator(const std::string& key, const Eigen::Matrix2cd& gate, const uint32_t& index, const uint32_t& num_qubits)
    {
        /*
        Calculates the operator for a gate applied to a single qubit

        Args:
            key (string): key for caching the operator
            gate (matrix_c_d): gate to apply to the qubit
            index (uint32_t): index of the qubit (Starting with 0)
            num_qubits (uint32_t): number of qubits in the system

        Returns:
            gate_f (matrix_c_d): resulting operator
        */

        if (num_qubits == 1)
        {
            Eigen::MatrixXcd gate_f = gate;
            return gate_f;
        }

        if(!key.empty() && s_complex_cache.count(key))
        {
            return s_complex_cache[key];
        }
        else
        {
            uint8_t I_cnt_l = index;
            uint8_t I_cnt_r = num_qubits - index - 1;
            Eigen::Vector2d idendity_cnt(I_cnt_l, I_cnt_r);
            Eigen::MatrixXcd gate_f = tensor_operator(idendity_cnt, gate);

            s_complex_cache[key] = gate_f; 
            return gate_f;
        }
    }

    Eigen::MatrixXd // Real Operator
    get_single_operator(const std::string& key, const Eigen::Matrix2d& gate, const uint32_t& index, const uint32_t& num_qubits)
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
        {
            Eigen::MatrixXd gate_f = gate;
            return gate_f;
        }

        if(!key.empty() && s_real_cache.count(key))
        {
            return s_real_cache[key];
        }
        else
        {
            uint8_t I_cnt_l = index;
            uint8_t I_cnt_r = num_qubits - index - 1;
            Eigen::Vector2d idendity_cnt(I_cnt_l, I_cnt_r);
            Eigen::MatrixXd gate_f = tensor_operator(idendity_cnt, gate);

            s_real_cache[key] = gate_f; 
            return gate_f;
        }
    }

    // tensor_operator
    Eigen::MatrixXcd // Complex Operator
    tensor_operator(const Eigen::Vector2d& idendity_cnt, const Eigen::Matrix2cd& op)
    {
        /*
        Performs the kronecker product of all the matrices in the given vector

        Args:
            idendity_cnt (Eigen::Vector2d):
            op (vector<matrix_c_d>): Gate

        Returns:
            op (matrix_c_d): result of kronecker products
        */
        uint32_t dim_I_l = idendity_cnt[0]*idendity_cnt[0];
        uint32_t dim_I_r = idendity_cnt[1]*idendity_cnt[1];

        if (dim_I_l != 0)
        {
            Eigen::MatrixXcd I_l = Eigen::MatrixXcd::Identity(dim_I_l, dim_I_l);
            Eigen::MatrixXcd op = kron_idendity_left(I_l, op);
        }
        if (dim_I_r != 0)
        {
            Eigen::MatrixXcd I_r = Eigen::MatrixXcd::Identity(dim_I_r, dim_I_r);
            Eigen::MatrixXcd op = kron_idendity_right(op, I_r);
        }
        return op;
    }

    Eigen::MatrixXd // real Operator
    tensor_operator(const Eigen::Vector2d& idendity_cnt, const Eigen::Matrix2d& op)
    {
        /*
        Performs the kronecker product of all the matrices in the given vector

        Args:
            op (vector<matrix_c_d>): vector of matrices

        Returns:
            res (matrix_c_d): result of kronecker products
        */
        uint32_t dim_I_l = idendity_cnt[0]*idendity_cnt[0];
        uint32_t dim_I_r = idendity_cnt[1]*idendity_cnt[1];

        if (dim_I_l != 0)
        {
            Eigen::MatrixXd I_l = Eigen::MatrixXd::Identity(dim_I_l, dim_I_l);
            Eigen::MatrixXd op = kron_idendity_left(I_l, op);
        }
        if (dim_I_r != 0)
        {
            Eigen::MatrixXd I_r = Eigen::MatrixXd::Identity(dim_I_r, dim_I_r);
            Eigen::MatrixXd op = kron_idendity_right(op, I_r);
        }
        return op;
    }

    Eigen::MatrixXcd tensor_operator(const std::vector<std::string>& keys) {
        /*
        Performs the kronecker product of all the matrices in the given vector referenced by the keys

        Args:
            keys (vector<string>): vector of matrix keys

        Returns:
            res (matrix_c_d): result of kronecker products
        */
        Eigen::MatrixXcd res = kron(*_matrixMap.at(keys[0]), *_matrixMap.at(keys[1]));
        for (uint32_t i = 2; i < keys.size(); i++) 
            res = kron(res, *_matrixMap.at(keys[i]));
        return res; 
    }
}