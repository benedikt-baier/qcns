#include "qcns_cpp/linalg/matrix_utils.h"
#include <unsupported/Eigen/KroneckerProduct>

namespace linalg {

    /*
    Add Functions
    */
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

    /*
    Multiply Functions
    */

    void multi_dot(const Eigen::MatrixXcd& transform_mat, Eigen::MatrixXcd& target_mat)
    {
        /*
        Complex Case
        ------------
        Performs the multi-dot product of op and state, op * state * op^H

        Args:
            op_1 (matrix_c_d): first matrix
            state (matrix_c_d): second matrix

        Returns:
            res (matrix_c_d): multi-dot product of op_1 and op_2 
        */
        Eigen::MatrixXcd transform_mat_H = transform_mat.adjoint();
        target_mat = transform_mat * target_mat * transform_mat_H;
    }

    void multi_dot(const Eigen::MatrixXd& transform_mat, Eigen::MatrixXd& target_mat)
    {
        /*
        Real Case
        ---------
        Performs the multi-dot product of op and state, op * state * op^H

        Args:
            op_1 (matrix_c_d): first matrix
            state (matrix_c_d): second matrix

        Returns:
            res (matrix_c_d): multi-dot product of op_1 and op_2 
        */  
        Eigen::MatrixXd transform_mat_H = transform_mat.transpose();
        target_mat = transform_mat * target_mat * transform_mat_H;
    }
    /*
    Kronecker Product Functions
    */

    Eigen::MatrixXcd kron_idendity_left(const Eigen::MatrixXcd& I, const Eigen::Matrix2cd& mat)
    {
        /*
        Grenzfall I*I*I*G_2x2 complex
        */
        Eigen::MatrixXcd res = Eigen::MatrixXcd::Zero(I.rows() * mat.RowsAtCompileTime, I.cols() * mat.ColsAtCompileTime);

        for (int i = 0; i < I.rows(); i++)
        {
            res.block(i*mat.rows(), i*mat.cols(), mat.rows(), mat.cols()) = mat;
        }
        return res;
    }

    Eigen::MatrixXd kron_idendity_left(const Eigen::MatrixXd& I, const Eigen::Matrix2d& mat)
    {
        /*
        Grenzfall I*I*I*G_2x2 real
        */
        Eigen::MatrixXd res = Eigen::MatrixXd::Zero(I.rows() * mat.RowsAtCompileTime, I.cols() * mat.ColsAtCompileTime);

        for (int i = 0; i < I.rows(); i++)
        {
            res.block(i*mat.rows(), i*mat.cols(), mat.rows(), mat.cols()) = mat;
        }
        return res;
    }

    Eigen::MatrixXcd kron_idendity_left(const Eigen::MatrixXcd& I, const Eigen::MatrixXcd& mat)
    {
        /*
        Grenzfall I*I*I*G_nxn complex
        */
        Eigen::MatrixXcd res = Eigen::MatrixXcd::Zero(I.rows() * mat.rows(), I.cols() * mat.cols());

        for (int i = 0; i < I.rows(); i++)
        {
            res.block(i*mat.rows(), i*mat.cols(), mat.rows(), mat.cols()) = mat;
        }
        return res;
    }

    Eigen::MatrixXd kron_idendity_left(const Eigen::MatrixXd& I, const Eigen::MatrixXd& mat)
    {
        /*
        Grenzfall I*I*I*G_nxn real
        */
        Eigen::MatrixXd res = Eigen::MatrixXd::Zero(I.rows() * mat.rows(), I.cols() * mat.cols());

        for (int i = 0; i < I.rows(); i++)
        {
            res.block(i*mat.rows(), i*mat.cols(), mat.rows(), mat.cols()) = mat;
        }
        return res;
    }

    Eigen::MatrixXcd kron_idendity_right(const Eigen::Matrix2cd& mat, const Eigen::MatrixXcd& I)
    {
        /*
        Grenzfall: G_2x2 * I * I * I complex
        */
        Eigen::MatrixXcd res(I.rows() * mat.RowsAtCompileTime, I.cols() * mat.ColsAtCompileTime);
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                res.block(i*I.rows(), j*I.cols(), I.rows(), I.cols()) = I*mat(i,j);
            }
        }
        return res;
    }

    Eigen::MatrixXd kron_idendity_right(const Eigen::Matrix2d& mat, const Eigen::MatrixXd& I)
    {
        /*
        Grenzfall: G_2x2 * I * I * I real
        */
        Eigen::MatrixXd res(I.rows() * mat.RowsAtCompileTime, I.cols() * mat.ColsAtCompileTime);
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                res.block(i*I.rows(), j*I.cols(), I.rows(), I.cols()) = I*mat(i,j);
            }
        }
        return res;
    }

    Eigen::MatrixXcd kron_idendity_right(const Eigen::MatrixXcd& mat, const Eigen::MatrixXcd& I)
    {
        /*
        Grenzfall: M_nxn * I * I * I complex
        */
        Eigen::MatrixXcd res(I.rows() * mat.rows(), I.cols() * mat.cols());
        for (int i = 0; i < mat.rows(); ++i)
        {
            for (int j = 0; j < mat.cols(); ++j)
            {
                res.block(i*I.rows(), j*I.cols(), I.rows(), I.cols()) = I * mat(i, j);
            }
        }
        return res;
    }

    Eigen::MatrixXd kron_idendity_right(const Eigen::MatrixXd& mat, const Eigen::MatrixXd& I)
    {
        /*
        Grenzfall: M_nxn * I * I * I real
        */
        Eigen::MatrixXd res(I.rows() * mat.rows(), I.cols() * mat.cols());
        for (int i = 0; i < mat.rows(); ++i)
        {
            for (int j = 0; j < mat.cols(); ++j)
            {
                res.block(i*I.rows(), j*I.cols(), I.rows(), I.cols()) = I * mat(i, j);
            }
        }
        return res;
    }

    // Eigen::MatrixXd
    // initializeRightKronMatrix(const uint32_t& I_rows, const uint32_t& I_cols)
    //  {
    //     Eigen::MatrixXd matrix(I_rows*2, I_cols*2);
    //     for (int i = 0; i < I_rows; ++i)
    //     {
    //         for (int j = 0; j < I_rows; ++j)
    //         {
    //             matrix.block<2, 2>(i, j) << 1, 0, 0, 1;
    //         }
    //     }
    //     return matrix;
    // }

    template<typename A, typename B>
    Eigen::MatrixXcd kron(const Eigen::MatrixBase<A>& mat1, const Eigen::MatrixBase<B>& mat2) {
        return Eigen::kroneckerProduct(mat1, mat2).eval();
    }

}