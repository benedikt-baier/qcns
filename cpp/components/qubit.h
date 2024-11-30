#pragma once

#include <vector>
#include <complex>
#include <stdlib.h>

#include "linalg.h"

class QSystem;

class Qubit
{
    /*
    Represents a single qubit

    Members:
        _qsystem (QSystem*): pointer to the parent QSystem
        _index (uint32_t): index of the qubit in the qsystem

    Args:
        qsystem (QSystem*): pointer to the parent QSystem
        index (uint32_t): index of the qubit in the qsystem

    Returns:
        /
    */

public:
    uint32_t _index;
    QSystem* _qsystem;

    Qubit(QSystem* qsystem = nullptr, uint32_t index = 0);
    ~Qubit();

    void
        X();

    void
        Y();

    void
        Z();

    void
        H();

    void
        SX();

    void
        SY();

    void
        SZ();

    void
        K();

    void
        T();

    void
        iSX();
    
    void
        iSY();

    void
        iSZ();

    void
        iT();

    void
        iK();
        
    void
        Rx(double& theta);

    void
        Ry(double& theta);

    void
        Rz(double& theta);

    void
        PHASE(double& theta);

    void
        custom_gate(matrix_c_d& gate);

    void
        CNOT(Qubit* target);

    void
        CY(Qubit* target);

    void
        CZ(Qubit* target);

    void
        CH(Qubit* target);

    void
        CPHASE(Qubit* target, double& theta);

    void
        CU(Qubit* target, matrix_c_d& gate);

    void
        SWAP(Qubit* target);

    void
        TOFFOLI(Qubit* control, Qubit* target);

    void
        CCU(Qubit* control, Qubit* target, matrix_c_d& gate_s);

    void
        CSWAP(Qubit* target_1, Qubit* target_2);

    void
        bell_state(Qubit* target, const uint32_t& state);

    uint32_t
        measure(uint32_t basis=0);

    uint32_t
        bsm(Qubit* target);

    double
        fidelity(matrix_c_d& state);
};

class QSystem
{
    /*
    Represents a qsystem with multiple qubits in it

    Members:
        _num_qubits (uint32_t): number of qubits in the system
        _qubits (vector<Qubit*>): qubits in the system
        _state (matrix_c_d): state of the system

    Args:
        num_qubits (uint32_t): number of qubits in the system

    Returns:
        /
    */

public:
    uint32_t _num_qubits;
    std::vector<Qubit*> _qubits;
    matrix_c_d _state;

    QSystem(const uint32_t& _num_qubits = 1);
    ~QSystem();
    Qubit* qubit(const uint32_t& index);
};

QSystem::QSystem(const uint32_t& num_qubits)
{
    /*
    Instantiates a qsystem

    Args:
        num_qubits (uint32_t): number of qubits in the system

    Returns:
        /
    */

    _num_qubits = num_qubits;
    _qubits.resize(_num_qubits);
    for (uint32_t i = 0; i < _num_qubits; i++)
        _qubits[i] = new Qubit(this, i);
    if (_num_qubits == 1)
    {
        _state = _P0;
    }
    else
    {
        std::vector<matrix_c_d> op(_num_qubits, _P0);
        _state = tensor_operator(op);
    }
}

QSystem::~QSystem()
{
    /*
    Destructs a Qsystem

    Args:
        /

    Returns:
        /
    */

    for(uint32_t i = 0; i < _num_qubits; i++)
        delete _qubits[i];
}

Qubit*
QSystem::qubit(const uint32_t& index)
{
    /*
    Accesses a single qubit in the system given an index

    Args:
        index (uint32_t): index of the qubit

    Returns:
        qubit (Qubit): qubit with the given index
    */

    return _qubits[index];
}

Qubit::Qubit(QSystem* qsystem, uint32_t index)
{
    /*
    Instantiates a qubit

    Args:
        qsystem (QSystem*): pointer to the parent QSystem
        index (uint32_t): index of the qubit in the qsystem

    Returns:
        /
    */

    _index = index;
    _qsystem = qsystem;
}

Qubit::~Qubit()
{
    /*
    Destructs a Qubit

    Args:
        /

    Returns:
        /
    */
}

void
Qubit::X()
{
    /*
    Applys a X gate to the qubit

    Args:
        /

    Returns:
        /
    */

    std::string key = "x_" + std::to_string(_qsystem->_num_qubits) + "_" + std::to_string(_index);
    matrix_c_d gate = get_single_operator(key, _X, _index, _qsystem->_num_qubits);
    multi_dot(gate, _qsystem->_state);
}

void
Qubit::Y()
{
    /*
    Applys a Y gate to the qubit

    Args:
        /

    Returns:
        /
    */

    std::string key = "y_" + std::to_string(_qsystem->_num_qubits) + "_" + std::to_string(_index);
    matrix_c_d gate = get_single_operator(key, _Y, _index, _qsystem->_num_qubits);
    multi_dot(gate, _qsystem->_state);
}

void
Qubit::Z()
{
    /*
    Applys a Z gate to the qubit

    Args:
        /

    Returns:
        /
    */

    std::string key = "z_" + std::to_string(_qsystem->_num_qubits) + "_" + std::to_string(_index);
    matrix_c_d gate = get_single_operator(key, _Z, _index, _qsystem->_num_qubits);
    multi_dot(gate, _qsystem->_state);
}

void
Qubit::H()
{
    /*
    Applys a H gate to the qubit

    Args:
        /

    Returns:
        /
    */

    std::string key = "h_" + std::to_string(_qsystem->_num_qubits) + "_" + std::to_string(_index);
    matrix_c_d gate = get_single_operator(key, _H, _index, _qsystem->_num_qubits);
    multi_dot(gate, _qsystem->_state);
}

void
Qubit::SX()
{
    /*
    Applys a square root X gate to the qubit

    Args:
        /

    Returns:
        /
    */

    std::string key = "sx_" + std::to_string(_qsystem->_num_qubits) + "_" + std::to_string(_index);
    matrix_c_d gate = get_single_operator(key, _SX, _index, _qsystem->_num_qubits);
    multi_dot(gate, _qsystem->_state);
}

void
Qubit::SY()
{
    /*
    Applys a square root Y gate to the qubit

    Args:
        /

    Returns:
        /
    */

    std::string key = "sy_" + std::to_string(_qsystem->_num_qubits) + "_" + std::to_string(_index);
    matrix_c_d gate = get_single_operator(key, _SY, _index, _qsystem->_num_qubits);
    multi_dot(gate, _qsystem->_state);
}

void
Qubit::SZ()
{
    /*
    Applys a square root X gate to the qubit

    Args:
        /

    Returns:
        /
    */

    std::string key = "sz_" + std::to_string(_qsystem->_num_qubits) + "_" + std::to_string(_index);
    matrix_c_d gate = get_single_operator(key, _SZ, _index, _qsystem->_num_qubits);
    multi_dot(gate, _qsystem->_state);
}

void
Qubit::K()
{
    /*
    Applys a K gate to the qubit

    Args:
        /

    Returns:
        /
    */

    std::string key = "k_" + std::to_string(_qsystem->_num_qubits) + "_" + std::to_string(_index);
    matrix_c_d gate = get_single_operator(key, _K, _index, _qsystem->_num_qubits);
    multi_dot(gate, _qsystem->_state);
}

void
Qubit::T()
{
    /*
    Applys a T gate to the qubit

    Args:
        /

    Returns:
        /
    */

    std::string key = "t_" + std::to_string(_qsystem->_num_qubits) + "_" + std::to_string(_index);
    matrix_c_d gate = get_single_operator(key, _T, _index, _qsystem->_num_qubits);
    multi_dot(gate, _qsystem->_state);
}

void
Qubit::iSX()
{
    /*
    Applys a square root X gate to the qubit

    Args:
        /

    Returns:
        /
    */

    std::string key = "isx_" + std::to_string(_qsystem->_num_qubits) + "_" + std::to_string(_index);
    matrix_c_d gate = get_single_operator(key, _iSX, _index, _qsystem->_num_qubits);
    multi_dot(gate, _qsystem->_state);
}

void
Qubit::iSY()
{
    /*
    Applys a square root X gate to the qubit

    Args:
        /

    Returns:
        /
    */

    std::string key = "isy_" + std::to_string(_qsystem->_num_qubits) + "_" + std::to_string(_index);
    matrix_c_d gate = get_single_operator(key, _SY, _index, _qsystem->_num_qubits);
    multi_dot(gate, _qsystem->_state);
}

void
Qubit::iSZ()
{
    /*
    Applys a square root X gate to the qubit

    Args:
        /

    Returns:
        /
    */

    std::string key = "isz_" + std::to_string(_qsystem->_num_qubits) + "_" + std::to_string(_index);
    matrix_c_d gate = get_single_operator(key, _iSZ, _index, _qsystem->_num_qubits);
    multi_dot(gate, _qsystem->_state);
}

void
Qubit::iT()
{
    /*
    Applys a T* gate to the qubit

    Args:
        /

    Returns:
        /
    */

    std::string key = "it_" + std::to_string(_qsystem->_num_qubits) + "_" + std::to_string(_index);
    matrix_c_d gate = get_single_operator(key, _iT, _index, _qsystem->_num_qubits);
    multi_dot(gate, _qsystem->_state);
}

void
Qubit::iK()
{
    /*
    Applys a square root X gate to the qubit

    Args:
        /

    Returns:
        /
    */

    std::string key = "ik_" + std::to_string(_qsystem->_num_qubits) + "_" + std::to_string(_index);
    matrix_c_d gate = get_single_operator(key, _iK, _index, _qsystem->_num_qubits);
    multi_dot(gate, _qsystem->_state);
}

void
Qubit::Rx(double& theta)
{
    /*
    Applys a Rx gate to the qubit given an angle

    Args:
        theta (double): angle to rotate the qubit

    Returns:
        /
    */

    matrix_c_d gate_s(4);
    gate_s[0] = cos(theta / 2);
    gate_s[1] = -imag_u * sin(theta / 2);
    gate_s[2] = -imag_u * sin(theta / 2);
    gate_s[3] = cos(theta / 2);

    std::string key = "rx_" + std::to_string(theta) + "_" + std::to_string(_qsystem->_num_qubits) + "_" + std::to_string(_index);
    matrix_c_d gate = get_single_operator(key, gate_s, _index, _qsystem->_num_qubits);
    multi_dot(gate, _qsystem->_state);
}

void
Qubit::Ry(double& theta)
{
    /*
    Applys a Ry gate to the qubit given an angle

    Args:
        theta (double): angle to rotate the qubit

    Returns:
        /
    */

    matrix_c_d gate_s(4);
    gate_s[0] = cos(theta / 2);
    gate_s[1] = -sin(theta / 2);
    gate_s[2] = -sin(theta / 2);
    gate_s[3] = cos(theta / 2);

    std::string key = "ry_" + std::to_string(theta) + "_" + std::to_string(_qsystem->_num_qubits) + "_" + std::to_string(_index);
    matrix_c_d gate = get_single_operator(key, gate_s, _index, _qsystem->_num_qubits);
    multi_dot(gate, _qsystem->_state);
}

void
Qubit::Rz(double& theta)
{
    /*
    Applys a Rz gate to the qubit given an angle

    Args:
        theta (double): angle to rotate the qubit

    Returns:
        /
    */

    matrix_c_d gate_s(4);
    gate_s[0] = exp(-imag_u * (theta / 2));
    gate_s[1] = { 0, 0 };
    gate_s[2] = { 0, 0 };
    gate_s[3] = exp(imag_u * (theta / 2));

    std::string key = "rz_" + std::to_string(theta) + "_" + std::to_string(_qsystem->_num_qubits) + "_" + std::to_string(_index);
    matrix_c_d gate = get_single_operator(key, gate_s, _index, _qsystem->_num_qubits);
    multi_dot(gate, _qsystem->_state);
}

void
Qubit::PHASE(double& theta)
{
    /*
    Applys a Phase gate to the qubit given an angle

    Args:
        theta (double): angle to rotate the qubit

    Returns:
        /
    */

    matrix_c_d gate_s(4);
    gate_s[0] = { 1, 0 };
    gate_s[1] = { 0, 0 };
    gate_s[2] = { 0, 0 };
    gate_s[3] = exp(imag_u * theta);

    std::string key = "p_" + std::to_string(theta) + "_" + std::to_string(_qsystem->_num_qubits) + "_" + std::to_string(_index);
    matrix_c_d gate = get_single_operator(key, gate_s, _index, _qsystem->_num_qubits);
    multi_dot(gate, _qsystem->_state);
}

void
Qubit::custom_gate(matrix_c_d& gate_s)
{
    /*
    Applys a custom gate to the qubit given a gate

    Args:
        gate (matrix_c_d): gate to apply to the qubit

    Returns:
        /
    */

    if (gate_s.size() != 4)
        return;

    matrix_c_d gate_h(4);
    gate_h[0] = conj(gate_s[0]);
    gate_h[1] = conj(gate_s[2]);
    gate_h[2] = conj(gate_s[1]);
    gate_h[3] = conj(gate_s[3]);

    if (!allclose(_I, dot(gate_s, gate_h)))
        return;

    matrix_c_d gate = get_single_operator("", gate_s, _index, _qsystem->_num_qubits);
    multi_dot(gate, _qsystem->_state);
}

void
Qubit::CNOT(Qubit* target)
{
    /*
    Applys a CNOT gate to the target qubit given this qubit as the source qubit

    Args:
        target (Qubit): target qubit to apply the CNOT gate to

    Returns:
        /
    */

    std::string key = "x_" + std::to_string(target->_qsystem->_num_qubits) + "_" + std::to_string(_index) + "_" + std::to_string(target->_index);
    matrix_c_d gate = get_double_operator(key, _X, _index, target->_index, target->_qsystem->_num_qubits);
    multi_dot(gate, target->_qsystem->_state);
}

void
Qubit::CY(Qubit* target)
{
    /*
    Applys a CY gate to the target qubit given this qubit as the source qubit

    Args:
        target (Qubit): target qubit to apply the CY gate to

    Returns:
        /
    */

    std::string key = "y_" + std::to_string(target->_qsystem->_num_qubits) + "_" + std::to_string(_index) + "_" + std::to_string(target->_index);
    matrix_c_d gate = get_double_operator(key, _Y, _index, target->_index, target->_qsystem->_num_qubits);
    multi_dot(gate, target->_qsystem->_state);
}

void
Qubit::CZ(Qubit* target)
{
    /*
    Applys a CZ gate to the target qubit given this qubit as the source qubit

    Args:
        target (Qubit): target qubit to apply the CZ gate to

    Returns:
        /
    */

    std::string key = "z_" + std::to_string(target->_qsystem->_num_qubits) + "_" + std::to_string(_index) + "_" + std::to_string(target->_index);
    matrix_c_d gate = get_double_operator(key, _Z, _index, target->_index, target->_qsystem->_num_qubits);
    multi_dot(gate, target->_qsystem->_state);
}

void
Qubit::CH(Qubit* target)
{
    /*
    Applys a CH gate to the target qubit given this qubit as the source qubit

    Args:
        target (Qubit): target qubit to apply the CH gate to

    Returns:
        /
    */

    std::string key = "h_" + std::to_string(target->_qsystem->_num_qubits) + "_" + std::to_string(_index) + "_" + std::to_string(target->_index);
    matrix_c_d gate = get_double_operator(key, _H, _index, target->_index, target->_qsystem->_num_qubits);
    multi_dot(gate, target->_qsystem->_state);
}

void
Qubit::CPHASE(Qubit* target, double& theta)
{
    /*
    Applys a CPhase gate to the target qubit given this qubit as the source qubit and an angle

    Args:
        target (Qubit): target qubit to apply the CPhase gate to
        theta (double): angle to rotate the target qubit

    Returns:
        /
    */

    matrix_c_d gate_s(4);
    gate_s[0] = { 1, 0 };
    gate_s[1] = { 0, 0 };
    gate_s[2] = { 0, 0 };
    gate_s[3] = exp(imag_u * theta);

    std::string key = "p_" + std::to_string(theta) + "_" + std::to_string(target->_qsystem->_num_qubits) + "_" + std::to_string(_index) + "_" + std::to_string(target->_index);
    matrix_c_d gate = get_double_operator(key, gate_s, _index, target->_index, target->_qsystem->_num_qubits);
    multi_dot(gate, target->_qsystem->_state);
}

void
Qubit::CU(Qubit* target, matrix_c_d& gate_s)
{
    /*
    Applys a CU (controlled custom unitary) gate to the target qubit given this qubit as the source qubit and an gate

    Args:
        target (Qubit): target qubit to apply the CU gate to
        gate_s (matrix_c_d): angle to rotate the target qubit

    Returns:
        /
    */

    if (gate_s.size() != 4)
        return;

    matrix_c_d gate_h(4);
    gate_h[0] = conj(gate_s[0]);
    gate_h[1] = conj(gate_s[2]);
    gate_h[2] = conj(gate_s[1]);
    gate_h[3] = conj(gate_s[3]);

    if (!allclose(_I, dot(gate_s, gate_h)))
        return;

    matrix_c_d gate = get_double_operator("", gate_s, _index, target->_index, target->_qsystem->_num_qubits);
    multi_dot(gate, target->_qsystem->_state);
}

void
Qubit::SWAP(Qubit* target)
{
    /*
    Swaps two qubits in a system by applying three CNOT gates

    Args:
        target (Qubit): target qubit to apply the Swap gate to

    Returns:
        /
    */

    target->CNOT(this);
    this->CNOT(target);
    target->CNOT(this);
}

void
Qubit::TOFFOLI(Qubit* control, Qubit* target)
{
    /*
    Applys the TOFFOLI gate to three qubits, with this qubit as the first control and a control and target qubit

    Args:
        control (Qubit): second control qubit to apply the TOFFOLI gate to
        target (Qubit): target qubit to apply the TOFFOLI gate to

    Returns:
        /
    */

    std::string key = "x_" + std::to_string(target->_qsystem->_num_qubits) + "_" + std::to_string(_index) + "_"  + std::to_string(control->_index) + "_" + std::to_string(target->_index);
    matrix_c_d gate = get_triple_operator(key, _X, _index, control->_index, target->_index, target->_qsystem->_num_qubits);
    multi_dot(gate, target->_qsystem->_state);
}

void
Qubit::CCU(Qubit* control, Qubit* target, matrix_c_d& gate_s)
{
    /*
    Applys a CCU gate to three qubits, with this qubit as the first control and a control and target qubit

    Args:
        control (Qubit): second control qubit to apply the CCU gate to
        target (Qubit): target qubit to apply the CCU gate to

    Returns:
        /
    */

    if (gate_s.size() != 4)
        return;

    matrix_c_d gate_h(4);
    gate_h[0] = conj(gate_s[0]);
    gate_h[1] = conj(gate_s[2]);
    gate_h[2] = conj(gate_s[1]);
    gate_h[3] = conj(gate_s[3]);

    if (!allclose(_I, dot(gate_s, gate_h)))
        return;

    matrix_c_d gate = get_triple_operator("", gate_s, _index, control->_index, target->_index, target->_qsystem->_num_qubits);
    multi_dot(gate, target->_qsystem->_state);
}

void
Qubit::CSWAP(Qubit* target_1, Qubit* target_2)
{
    /*
    Applys a CSwap gate to three qubits with this qubit as the first control and two target qubits with three TOFFOLI gates

    Args:
        target_1 (Qubit): first target qubit to apply the CSWAP gate to
        target_2 (Qubit): second target qubit to apply the CSWAP gate to

    Returns:
        /
    */

    this->TOFFOLI(target_2, target_1);
    this->TOFFOLI(target_1, target_2);
    this->TOFFOLI(target_2, target_1);
}

void
Qubit::bell_state(Qubit* target, const uint32_t& state)
{
    /*
    Transforms the state into a bell state

    Args:
        target (Qubit): target qubit
        bell_state (uint32_t): bell state to transform state into

    Returns:
        /
    */

    if (state > 3)
        return;
    if (state < 0)
        return;

    std::string key = "bs_" + std::to_string(state) + "_" + std::to_string(target->_qsystem->_num_qubits) + "_" + std::to_string(_index) + "_" + std::to_string(target->_index);
    matrix_c_d gate = get_bell_operator(key, state, _index, target->_index, target->_qsystem->_num_qubits);
    multi_dot(gate, target->_qsystem->_state);
}

uint32_t
Qubit::measure(uint32_t basis)
{
    /*
    Measures the given qubit

    Args:
        /

    Returns:
        res (uint32_t): measurement result (either 0 or 1)
    */

    if(basis == 1)
        this->H();

    if(basis == 2)
    {
        this->iSZ();
        this->H();
    }

    std::string key = "m0_" + std::to_string(_qsystem->_num_qubits) + "_" + std::to_string(_index);
    matrix_c_d measure = get_single_operator(key, _P0, _index, _qsystem->_num_qubits);
    matrix_c_d tmp = dot(measure, _qsystem->_state);
    double prob = trace(tmp);
    double random_num = ((double)rand() / (RAND_MAX));

    multi_dot(measure, _qsystem->_state);
    multiply(_qsystem->_state, 1 / prob);
    return 0;

    // if (random_num <= prob)
    // {
    //     multi_dot(measure, _qsystem->_state);
    //     multiply(_qsystem->_state, 1 / prob);
    //     return 0;
    // }
    // else
    // {
    //     key = "m1_" + std::to_string(_qsystem->_num_qubits) + "_" + std::to_string(_index);
    //     measure = get_single_operator(key, _P1, _index, _qsystem->_num_qubits);
    //     multi_dot(measure, _qsystem->_state);
    //     multiply(_qsystem->_state, 1 / (1 - prob));
    //     return 1;
    // }
}

uint32_t
Qubit::bsm(Qubit* target)
{
    /*
    Performs a Bell state measurement (bsm) on this qubit and the target qubit

    Args:
        target (Qubit): target qubit

    Returns:
        res (uint32_t): measurement result
    */

    std::string key = "bsm_" + std::to_string(target->_qsystem->_num_qubits) + "_" + std::to_string(_index) + "_" + std::to_string(target->_index);
    matrix_c_d gate = get_bsm_operator(key, _index, target->_index, target->_qsystem->_num_qubits);
    multi_dot(gate, target->_qsystem->_state);

    return 2 * this->measure() + target->measure();
}

double
Qubit::fidelity(matrix_c_d& state)
{
    /*
    Computes the fidelity between the state of this qubit and a given state

    Args:
        state (matrix_c_d): state to compare to

    Returns:
        fidelity (double): fidelity between the two states
    */

    matrix_c_d tmp = dot(this->_qsystem->_state, state);

    return trace(tmp) + real(std::sqrt(det(this->_qsystem->_state) * det(state)));
}

void
ptrace(Qubit* qubit)
{
    /*
    Traces out the given qubit

    Args:
        qubit (Qubit*): qubit

    Returns:
        /
    */

    uint32_t idx = qubit->_index;
    uint32_t num_qubits = qubit->_qsystem->_num_qubits;

    uint32_t offset_i = (1 << (num_qubits));
    uint32_t offset_k = ((1 << num_qubits) + 1) * (1 << (num_qubits - idx - 1));

    idx_array indices_j = generate_trace_indices(idx, num_qubits);

    idx_array indices_i(1 << (num_qubits - 1), 0);
    for (uint32_t i = 0; i < (1u << (num_qubits - 1)); i++)
        indices_i[i] = offset_i * indices_j[i];

    matrix_c_d array_traced(1 << (2 * (num_qubits - 1)), {0, 0});

    uint32_t counter = 0;
    for (uint32_t i : indices_i)
    {
        for (uint32_t j : indices_j)
        {
            array_traced[counter] += qubit->_qsystem->_state[i + j];
            array_traced[counter] += qubit->_qsystem->_state[offset_k + i + j];
            counter++;
        }
    }

    qubit->_qsystem->_state = array_traced;
    qubit->_qsystem->_num_qubits--;

    qubit->_qsystem->_qubits[idx] = nullptr;

    for(uint32_t i = idx; i < qubit->_qsystem->_num_qubits; i++)
    {
        qubit->_qsystem->_qubits[i] = qubit->_qsystem->_qubits[i + 1];
        qubit->_qsystem->_qubits[i]->_index = i;
    }

    qubit->_qsystem->_qubits.resize(qubit->_qsystem->_num_qubits);

    delete qubit;
}

void
partial_trace(std::vector<Qubit*> qubit_l)
{
    /*
    Traces out the given qubits

    Args:
        qubit_l (vector<Qubit*>): vector of qubits

    Returns:
        /
    */

    for (Qubit* q : qubit_l)
        ptrace(q);
}