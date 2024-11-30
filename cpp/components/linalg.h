#pragma once

#include <complex>
#include <vector>
#include <unordered_map>
#include <iostream>

typedef std::vector<std::complex<double>> matrix_c_d;
typedef std::vector<uint32_t> idx_array;

static const std::complex<double> imag_u = { 0, 1 };
static const matrix_c_d _I{ {{1, 0}, {0, 0}, {0, 0}, {1, 0}} };
static const matrix_c_d _P0{ {{1, 0}, {0, 0}, {0, 0}, {0, 0}} };
static const matrix_c_d _P1{ {{0, 0}, {0, 0}, {0, 0}, {1, 0}} };
static const matrix_c_d _X{ {{0, 0}, {1, 0}, {1, 0}, {0, 0}} };
static const matrix_c_d _Y{ {{0, 0}, {0, -1}, {0, 1}, {0, 0}} };
static const matrix_c_d _Z{ {{1, 0}, {0, 0}, {0, 0}, {-1, 0}} };
static const matrix_c_d _H{ {{0.70710677, 0}, {0.70710677, 0}, {0.70710677, 0}, {-0.70710677, 0}} };
static const matrix_c_d _K{ {{0.5, 0.5}, {0.5, -0.5}, {-0.5, 0.5}, {-0.5, -0.5}} };
static const matrix_c_d _SX{ {{0.5, 0.5}, {0.5, -0.5}, {0.5, -0.5}, {0.5, 0.5}} };
static const matrix_c_d _SY{ {{0.5, 0.5}, {-0.5, -0.5}, {0.5, 0.5}, {0.5, 0.5}} };
static const matrix_c_d _SZ{ {{1, 0}, {0, 0}, {0, 0}, {0, 1}} };
static const matrix_c_d _T{ {{1, 0}, {0, 0}, {0, 0}, exp(0.785398 * imag_u)} };
static const matrix_c_d _iSX{{{0.5, -0.5}, {0.5, 0.5}, {0.5, 0.5}, {0.5, -0.5}}};
static const matrix_c_d _iSY{{{0.5, -0.5}, {0.5, -0.5}, {-0.5, 0.5}, {0.5, -0.5}}};
static const matrix_c_d _iSZ{ {{1, 0}, {0, 0}, {0, 0}, {0, -1}} };
static const matrix_c_d _iT{ {{1, 0}, {0, 0}, {0, 0}, exp(0.785398 * (-imag_u))} };
static const matrix_c_d _iK{{{0.5, -0.5}, {-0.5, -0.5}, {0.5, 0.5}, {-0.5, 0.5}}};

std::unordered_map<std::string, matrix_c_d> s_cache;
std::unordered_map<std::string, matrix_c_d> d_cache;
std::unordered_map<std::string, matrix_c_d> t_cache;

matrix_c_d
add(const matrix_c_d& op_1, const matrix_c_d& op_2)
{
    /*
    Adds two matrices

    Args:
        op_1 (matrix_c_d): first matrix
        op_2 (matrix_c_d): second matrix

    Returns:
        res (matrix_c_d): resulting matrix
    */

    matrix_c_d res(op_1.size(), { 0.0, 0.0 });
    for (uint32_t i = 0; i < op_1.size(); i++)
        res[i] = op_1[i] + op_2[i];

    return res;
}

void
multiply(matrix_c_d& op, const std::complex<double>& scalar)
{
    /*
    Multiplies a matrix with a scalar

    Args:
        op (matrix_c_d): matrix
        scalar (complex<double>): scalar

    Returns:
        /
    */

    for (uint32_t i = 0; i < op.size(); i++)
        op[i] *= scalar;
}

bool
allclose(const matrix_c_d& op_1, const matrix_c_d& op_2, const double& tolerance = 1e-5)
{
    /*
    Checks if op_1 and op_2 are similar given a tolerance matrices have to be same size

    Args:
        op_1 (matrix_c_d): first matrix
        op_2 (matrix_c_d): second matrix
        tolerance (double): tolerance

    Returns:
        res (bool): true if op_1 and op_2 are similar false otherwise
    */

    for (uint32_t i = 0; i < op_1.size(); i++)
        if (abs(real(op_1[i]) - real(op_2[i])) > tolerance || abs(imag(op_1[i]) - imag(op_2[i])) > tolerance)
            return false;
    return true;
}

matrix_c_d
dot(const matrix_c_d& op_1, const matrix_c_d& op_2)
{
    /*
    Performs the dot product of op_1 and op_2

    Args:
        op_1 (matrix_c_d): first matrix
        op_2 (matrix_c_d): second matrix

    Returns:
        res (matrix_c_d): dot product of op_1 and op_2
    */

    matrix_c_d res(op_1.size(), { 0.0, 0.0 });
    uint32_t dim = (uint32_t)sqrt(op_1.size());

    for (uint32_t i = 0; i < dim; i++)
    {
        for (uint32_t k = 0; k < dim; k++)
        {
            if (op_1[i * dim + k] == 0.0)
                continue;
            for (uint32_t j = 0; j < dim; j++)
                if (op_2[k * dim + j] != 0.0)
                    res[i * dim + j] += op_1[i * dim + k] * op_2[k * dim + j];
        }
    }

    return res;
}

void
multi_dot(const matrix_c_d& op, matrix_c_d& state)
{
    /*
    Performs the multi-dot product of op and state, op * state * op^H

    Args:
        op_1 (matrix_c_d): first matrix
        state (matrix_c_d): second matrix

    Returns:
        res (matrix_c_d): multi-dot product of op_1 and op_2 
    */

    matrix_c_d res_tmp(op.size(), { 0.0, 0.0 });
    uint32_t dim = (uint32_t)sqrt(op.size());

    for (uint32_t i = 0; i < dim; i++)
    {
        for (uint32_t k = 0; k < dim; k++)
        {
            if (op[i * dim + k] == 0.0)
                continue;
            for (uint32_t j = 0; j < dim; j++)
                if (state[k * dim + j] != 0.0)
                    res_tmp[i * dim + j] += op[i * dim + k] * state[k * dim + j];
        }
    }

    for (uint32_t i = 0; i < state.size(); i++)
        state[i] = { 0.0, 0.0 };

    for (uint32_t i = 0; i < dim; i++)
    {
        for (uint32_t k = 0; k < dim; k++)
        {
            if (res_tmp[i * dim + k] == 0.0)
                continue;

            for (uint32_t j = 0; j < dim; j++)  
                if (op[j * dim + k] != 0.0)
                    state[i * dim + j] += res_tmp[i * dim + k] * conj(op[j * dim + k]);
        }
    }
}

matrix_c_d
kron(const matrix_c_d& op_1, const matrix_c_d& op_2)
{
    /*
    Calculates the kronecker product of of two matrices

    Args:
        op_1 (matrix_c_d): first matrix
        op_2 (matrix_c_d): second matrix

    Returns:
        res (matrix_c_d): kronecker product of op_1 and op_2
    */

    matrix_c_d res(op_1.size() * op_2.size(), { 0.0, 0.0 });
    uint32_t dim_op_1 = (uint32_t)sqrt(op_1.size());
    uint32_t dim_op_2 = (uint32_t)sqrt(op_2.size());

    for (uint32_t i = 0; i < dim_op_1; i++)
    {
        for (uint32_t j = 0; j < dim_op_1; j++)
        {
            if (op_1[i * dim_op_1 + j] == 0.0)
                continue;
            for (uint32_t k = 0; k < dim_op_2; k++)
                for (uint32_t l = 0; l < dim_op_2; l++)
                    if (op_2[k * dim_op_2 + l] != 0.0)
                        res[((i * dim_op_2 + k) * dim_op_1 + j) * dim_op_2 + l] = op_1[i * dim_op_1 + j] * op_2[k * dim_op_2 + l];
        }     
    }

    return res;
}

double
trace(matrix_c_d& op)
{
    /*
    Traces the given matrix, sum of diagonal elements

    Args:
        op (matrix_c_d): matrix to trace

    Returns:
        res (double): sum of diagonal elements
    */

    uint32_t dim = (uint32_t)sqrt(op.size());
    std::complex<double> sum = { 0, 0 };
    for (uint32_t i = 0; i < dim; i++)
        sum += op[i * (dim + 1)];

    return real(sum);
}

matrix_c_d
tensor_operator(std::vector<matrix_c_d>& op)
{
    /*
    Performs the kronecker product of all the matrices in the given vector

    Args:
        op (vector<matrix_c_d>): vector of matrices

    Returns:
        res (matrix_c_d): result of kronecker products
    */

    matrix_c_d res = kron(op[0], op[1]);
    for (uint32_t i = 2; i < op.size(); i++)
        res = kron(res, op[i]);

    return res;
}

matrix_c_d
get_single_operator(const std::string& key, const matrix_c_d& gate, const uint32_t& index, const uint32_t& num_qubits)
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

    if(!key.empty() && s_cache.count(key))
        return s_cache[key];

    std::vector<matrix_c_d> op(num_qubits, _I);
    op[index] = gate;
    matrix_c_d gate_f = tensor_operator(op);

    if(!key.empty())
        s_cache[key] = gate_f;

    return gate_f;
}

matrix_c_d
get_double_operator(const std::string& key, const matrix_c_d& gate, const uint32_t& c_index, const uint32_t& t_index, const uint32_t& t_num_qubits)
{
    /*
    Calculates the operator for a gate applied to a two qubits

    Args:
        key (string): key for caching the operator
        gate (matrix_c_d): gate to apply to the qubit
        c_index (uint32_t): index of the control qubit
        t_index (uint32_t): index of the target qubit
        t_num_qubits (uint32_t): number of qubits in the target system

    Returns:
        gate_f (matrix_c_d): resulting operator
    */

    if(!key.empty() && d_cache.count(key))
        return d_cache[key];

    std::string key_1 = "m0_" + std::to_string(t_num_qubits) + "_" + std::to_string(c_index);
    matrix_c_d proj0 = get_single_operator(key_1, _P0, c_index, t_num_qubits);
    std::vector<matrix_c_d> proj1(t_num_qubits, _I);
    proj1[c_index] = _P1;
    proj1[t_index] = gate;

    matrix_c_d gate_f = add(proj0, tensor_operator(proj1));

    if(!key.empty())
        d_cache[key] = gate_f;

    return gate_f;
}

matrix_c_d
get_triple_operator(const std::string& key, const matrix_c_d& gate, const uint32_t& c1_index, const uint32_t& c2_index, const uint32_t& t_index, const uint32_t& t_num_qubits)
{
    /*
    Calculates the operator for a gate applied to a three qubits

    Args:
        key (string): key for caching the operator
        gate (matrix_c_d): gate to apply to the qubit
        c1_index (uint32_t): index of the first control qubit
        c2_index (uint32_t): index of the second control qubit
        t_index (uint32_t): index of the target qubit
        t_num_qubits (uint32_t): number of qubits in the target system

    Returns:
        gate_f (matrix_c_d): resulting operator
    */

    if(!key.empty() && t_cache.count(key))
        return t_cache[key];

    uint32_t c1, c2;
    if (c1_index <= c2_index)
    {
        c1 = c1_index;
        c2 = c2_index;
    }
    else
    {
        c1 = c2_index;
        c2 = c1_index;
    }

    matrix_c_d CNOTijk;
    std::vector<matrix_c_d> op(t_num_qubits, _I);
    op[c1] = _P0;
    op[c2] = _P0;
    CNOTijk = tensor_operator(op);

    op.resize(t_num_qubits, _I);
    op[c1] = _P0;
    op[c2] = _P1;
    CNOTijk = add(CNOTijk, tensor_operator(op));

    op.resize(t_num_qubits, _I);
    op[c1] = _P1;
    op[c2] = _P0;
    CNOTijk = add(CNOTijk, tensor_operator(op));

    op.resize(t_num_qubits, _I);
    op[c1] = _P1;
    op[c2] = _P1;
    op[t_index] = gate;

    matrix_c_d gate_f = add(CNOTijk, tensor_operator(op));

    if(!key.empty())
        t_cache[key] = gate_f;

    return gate_f;
}

matrix_c_d
get_bell_operator(const std::string& key, const uint32_t& bell_state, const uint32_t& c_index, const uint32_t& t_index, const uint32_t& t_num_qubits)
{
    /*
    Generates the bell operator

    Args:
        key (std::string): key used for caching
        bell_state (uint32_t): bell state to generate
        c_index (uint32_t): index of control qubit
        t_index (uint32_t): index of target qubit
        t_num_qubits (uint32_t): number of qubits in target QSystem

    Returns:
        res (matrix_c_d): resulting tensor for the bell operator operator
    */

    if(!key.empty() && d_cache.count(key))
        return d_cache[key];

    std::string key_h = "h_" + std::to_string(t_num_qubits) + "_" + std::to_string(c_index);
    std::string key_x = "x_" + std::to_string(t_num_qubits) + "_" + std::to_string(c_index) + "_" + std::to_string(t_index);

    matrix_c_d cnot_gate = get_double_operator(key_x, _X, c_index, t_index, t_num_qubits);
    matrix_c_d h_gate = get_single_operator(key_h, _H, c_index, t_num_qubits);
    matrix_c_d gate_t = dot(cnot_gate, h_gate);

    if (bell_state == 0 && !key.empty())
    {
        d_cache[key] = gate_t;
        return gate_t;
    }
    else if(bell_state == 1 && !key.empty())
    {
        std::string key_x_t = "x_" + std::to_string(t_num_qubits) + "_" + std::to_string(t_index);
        matrix_c_d x_gate = get_single_operator(key_x_t, _X, t_index, t_num_qubits);
        matrix_c_d gate_f = dot(gate_t, x_gate);
        d_cache[key] = gate_f;
        return gate_f;
    }
    else if(bell_state == 2 && !key.empty())
    {
        std::string key_x_c = "x_" + std::to_string(t_num_qubits) + "_" + std::to_string(c_index);
        matrix_c_d x_gate = get_single_operator(key_x_c, _X, c_index, t_num_qubits);
        matrix_c_d gate_f = dot(gate_t, x_gate);
        d_cache[key] = gate_f;
        return gate_f;
    }
    else if(bell_state == 3 && !key.empty())
    {
        std::string key_x_c = "x_" + std::to_string(t_num_qubits) + "_" + std::to_string(c_index);
        std::string key_x_t = "x_" + std::to_string(t_num_qubits) + "_" + std::to_string(t_index);

        matrix_c_d x_gate_t = get_single_operator(key_x_t, _X, t_index, t_num_qubits);
        matrix_c_d x_gate_c = get_single_operator(key_x_c, _X, c_index, t_num_qubits);

        matrix_c_d gate_s = dot(x_gate_t, x_gate_c);
        matrix_c_d gate_f = dot(gate_t, gate_s);

        d_cache[key] = gate_f;
        return gate_f;
    }
    return get_single_operator("", _I, 0, t_num_qubits);
}

matrix_c_d
get_bsm_operator(const std::string& key, const uint32_t& c_index, const uint32_t& t_index, const uint32_t& t_num_qubits)
{
    /*
    Generates the bsm operator
    
    Args:
        key (std::string): key used for caching
        c_index (uint32_t): index of control qubit
        t_index (uint32_t): index of target qubit
        t_num_qubits (uint32_t): number of qubits in target QSystem
        
    Returns:
        res (matrix_c_d): resulting tensor for the bsm operator
    */

    if(!key.empty() && d_cache.count(key))
        return d_cache[key];

    std::string key_h = "h_" + std::to_string(t_num_qubits) + "_" + std::to_string(c_index);
    std::string key_x = "x_" + std::to_string(t_num_qubits) + "_" + std::to_string(c_index) + "_" + std::to_string(t_index);

    matrix_c_d cnot_gate = get_double_operator(key_x, _X, c_index, t_index, t_num_qubits);
    matrix_c_d h_gate = get_single_operator(key_h, _H, c_index, t_num_qubits);
    matrix_c_d gate_f = dot(h_gate, cnot_gate);

    if(!key.empty())
        d_cache[key] = gate_f;

    return gate_f;
}

idx_array inline
generate_trace_indices(const uint32_t& idx, const uint32_t& num_qubits)
{
    /*
    Generates the trace indices in a matrix for tracing out specific qubits

    Args:
        idx (uint32_t): index to trace out
        num_qubits (uint32_t): number of qubits

    Returns:
        flatten_list (vector<uint32_t>): trace indices
    */

    uint32_t num_indices = 1 << num_qubits;
    uint32_t num_sub_indices = 1 << (num_qubits - idx - 1);
    uint32_t num_sub_indices_1 = 1 << (idx + 1);

    idx_array all_indices(num_indices, 0);
    for (uint32_t i = 0; i < num_indices; i++)
        all_indices[i] = i;

    std::vector<idx_array> nested_list(1 << idx);
    uint32_t counter = 0;
    for (uint32_t i = 0; i < num_sub_indices_1; i += 2)
    {
        idx_array sub_array(num_sub_indices);
        for (uint32_t j = 0; j < num_sub_indices; j++)
            sub_array[j] = all_indices[i * num_sub_indices + j];
        nested_list[counter] = sub_array;
        counter++;
    }

    idx_array flatten_list;
    flatten_list.reserve(1 << (num_qubits - 1));
    for (uint32_t i = 0; i < (1u << idx); i++)
        for (uint32_t j = 0; j < num_sub_indices; j++)
            flatten_list[i * num_sub_indices + j] = nested_list[i][j];

    return flatten_list;
}

std::complex<double>
det(matrix_c_d& state)
{
    uint32_t index, size = (uint32_t)std::sqrt(state.size());
    std::complex<double> num_1, num_2, det = 1, total = 1;
    matrix_c_d op = state;
    matrix_c_d tmp(size, {0, 0});

    for(uint32_t i = 0; i < size; i++)
    {
        index = i;

        while(index < size && op[index * size + i] == 0.)
            index++;

        if(index == size)
            continue;
        if(index != i)
        {
            for(uint32_t j = 0; j < size; j++)
                std::swap(op[index * size + j], op[i * size + j]);
            if((index - i) % 2 == 1)
                det = -det;
        }
        for (uint32_t j = 0; j < size; j++)
            tmp[j] = op[i * size + j];

        for (uint32_t j = i + 1; j < size; j++)
        {
            num_1 = tmp[i];
            num_2 = op[j * size + i];
 
            for (uint32_t k = 0; k < size; k++)
                op[j * size + k] = (num_1 * op[j * size + k]) - (num_2 * op[k]);
            total = total * num_1;
        }
    }

    for (uint32_t i = 0; i < size; i++)
        det = det * op[i * size + i];

    return (det / total);
}

void
print(const matrix_c_d& matrix)
{
    /*
    Prints a matrix

    Args:
        matrix (matrix_c_d): matrix to print

    Returns:
        /
    */

    uint32_t dim = (uint32_t)sqrt(matrix.size());
    for (uint32_t i = 0; i < dim; i++)
    {
        for (uint32_t j = 0; j < dim; j++)
            std::cout << real(matrix[i * dim + j]) << " + " << imag(matrix[i * dim + j]) << "j" << " ";
        std::cout << std::endl;
    }
}
