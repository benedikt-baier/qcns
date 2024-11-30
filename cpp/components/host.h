#pragma once

#include <string>
#include <vector>
#include <map>
#include <thread>
#include <iterator>

#include "qubit.h"
#include "channels.h"

const double single_gate_time = 1e-6;
const double double_gate_time = 2e-6;
const double triple_gate_time = 3e-6;
const double measure_time = 4e-6;
const double bsm_time = 1e-5;


class Simulation;

class Host
{
    /*
    Represents a Host in a network connected to other hosts via channels
    Each Host runs in a different thread

    Members:
        q_mem (std::vector<Qubit>): quantum memory, used to store qubits
        c_mem (std::vector<std::string>): classical memory, used to store classical messages

        _thread (std::thread): thread used to execute custom function in the Host

        q_channels_out (std::map<uint32_t, std::shared_ptr<QChannel>>): quantum channels used for sending qubits
        q_channels_in (std::map<uint32_t, std::shared_ptr<QChannel>>): quantum channels used for receiving qubits

        c_channels_out (std::map<uint32_t, std::shared_ptr<CChannel>>): classical channels used for sending classical messages
        c_channels_in (std::map<uint32_t, std::shared_ptr<CChannel>>): classical channels used for receiving classical messages

        _id (uint32_t): custom id of the host
        _sim (Simulation*): pointer to the sim object, used for creating sending and receiving thread-safe qubits
    */

public:

    std::thread _thread;

    std::map<uint32_t, std::shared_ptr<QChannel>> q_channels_out;
    std::map<uint32_t, std::shared_ptr<QChannel>> q_channels_in;

    std::map<uint32_t, std::shared_ptr<CChannel>> c_channels_out;
    std::map<uint32_t, std::shared_ptr<CChannel>> c_channels_in;

    uint32_t _id;
    Simulation* _sim;
    bool _stop_event = false;
    bool _request_waiting = false;
    double _time = 0.0;

    Host(uint32_t id, Simulation* sim);
    ~Host();

    void
        set_qconnection(Host* host, const double& length = 0.0, const std::vector<Error*>& errors = {});

    void
        set_cconnection(Host* host, const double& length = 0.0);

    void
        set_connection(Host* host, const double& length=0.0, const std::vector<Error*>& errors = {});

    QSystem*
        create_qsystem(const uint32_t& num_qubits);

    void
        delete_qsystem(QSystem*& qsys);

    void
        send_qubit(const uint32_t& host, Qubit* qubit);

    Qubit*
        receive_qubit(const uint32_t& host);

    Qubit*
        receive_qubit_wait(const uint32_t& host, uint32_t timeout=1);

    void
        send_classical(const uint32_t& host, std::string message);

    std::string
        receive_classical(const uint32_t& host);

    std::string
        receive_classical_wait(const uint32_t& host, uint32_t timeout=1);

    void
        X(Qubit* q);

    void
        Y(Qubit* q);

    void
        Z(Qubit* q);

    void
        H(Qubit* q);

    void
        SX(Qubit* q);

    void
        SY(Qubit* q);

    void
        SZ(Qubit* q);

    void
        K(Qubit* q);

    void
        T(Qubit* q);

    void
        iSX(Qubit* q);

    void
        iSY(Qubit* q);

    void
        iSZ(Qubit* q);

    void
        iT(Qubit* q);

    void
        iK(Qubit* q);

    void
        Rx(Qubit* q, double& theta);

    void
        Ry(Qubit* q, double& theta);

    void
        Rz(Qubit* q, double& theta);

    void
        PHASE(Qubit* q, double& theta);

    void
        custom_gate(Qubit* q, matrix_c_d& gate);

    void
        CNOT(Qubit* q, Qubit* target);

    void
        CY(Qubit* q, Qubit* target);

    void
        CZ(Qubit* q, Qubit* target);

    void
        CH(Qubit* q, Qubit* target);

    void
        CPHASE(Qubit* q, Qubit* target, double& theta);

    void
        CU(Qubit* q, Qubit* target, matrix_c_d& gate);

    void
        SWAP(Qubit* q, Qubit* target);

    void
        TOFFOLI(Qubit* q, Qubit* control, Qubit* target);

    void
        CCU(Qubit* q, Qubit* control, Qubit* target, matrix_c_d& gate_s);

    void
        CSWAP(Qubit* q, Qubit* target_1, Qubit* target_2);

    void
        bell_state(Qubit* q, Qubit* target, const uint32_t& state);

    uint32_t
        measure(Qubit* q);

    uint32_t
        bsm(Qubit* q, Qubit* target);

    virtual void
        start();

    void
        join();
};

class Simulation
{
    /*
    Represents a Simulation, starts all hosts as different threads and executes their custom functions, responsible for creating and deleting thread-safe qubits

    Member:
        _hosts (std::vector<Host*>): collection of hosts in the Simulation
    */

public:

    std::vector<Host*> _hosts;
    std::vector<std::vector<QSystem*>> _qsys_l;
    std::vector<std::vector<Qubit*>> _qubits_l;
    std::vector<Host*> _host_l;
    mutable std::mutex _qsys_l_mutex;

    Simulation();
    ~Simulation();

    QSystem*
        create_qsystem(const uint32_t& num_qubits);

    void
        delete_qsystem(QSystem* qsystem);

    void
        combine_states(std::vector<QSystem*> qsys_l);

    void
        ptrace(Qubit* qubit);

    void
        partial_trace(std::vector<Qubit*> qubit_l);

    bool
        check_list(std::vector<QSystem*>& req);

    void
        update_requests();
    
    void
        add_request(Host* host, std::vector<Qubit*> q_l);

    void
        complete_request(Host* host);

    void
        add_host(Host* host);

    void
        add_hosts(std::vector<Host*> hosts);

    void
        run();

    void
        run_with_infinity(std::vector<Host*> hosts, std::vector<Host*> routers);
};


Host::Host(uint32_t id, Simulation* sim)
{
    /*
    Initialises a Host

    Args:
        id (uint32_t): custom id
        sim (Simulation*): pointer to the sim object, used for creating sending and receiving thread-safe qubits
    */

    _id = id;
    _sim = sim;
}

Host::~Host()
{
    /*
    Destructs a Host

    Args:
        /

    Returns:
        /
    */
}

void
Host::set_qconnection(Host* host, const double& length, const std::vector<Error*>& errors)
{
    /*
    Creates a bi-directional quantum connection between this host and another host

    Args:
        host (Host): other host to connect to
        errors (std::vector<Error*>): errors for the channels

    Returns:
        /
    */

    std::shared_ptr<QChannel> s_t_h = std::make_shared<QChannel>(length, errors);
    std::shared_ptr<QChannel> h_t_s = std::make_shared<QChannel>(length, errors);

    q_channels_out[host->_id] = s_t_h;
    q_channels_in[host->_id] = h_t_s;
    host->q_channels_out[_id] = h_t_s;
    host->q_channels_in[_id] = s_t_h;
}

void
Host::set_cconnection(Host* host, const double& length)
{
    /*
    Creates a bi-directional classical connection between this host and another host

    Args:
        host (Host): other host to connect to

    Returns:
        /
    */

    std::shared_ptr<CChannel> s_t_h = std::make_shared<CChannel>(length);
    std::shared_ptr<CChannel> h_t_s = std::make_shared<CChannel>(length);

    c_channels_out[host->_id] = s_t_h;
    c_channels_in[host->_id] = h_t_s;
    host->c_channels_out[_id] = h_t_s;
    host->c_channels_in[_id] = s_t_h;
}

void
Host::set_connection(Host* host, const double& length, const std::vector<Error*>& errors)
{
    /*
    Creates a bi-directional classical and quantum connection between this host and another host

    Args:
        host (Host): other host to connect to
        errors (std::vector<Error*>): errors for the quantum channels

    Returns:
        /
    */

    set_qconnection(host, length, errors);
    set_cconnection(host, length);
}

QSystem*
Host::create_qsystem(const uint32_t& num_qubits)
{
    /*
    Creates a QSystem

    Args:
        num_qubits (uint32_t): number of qubits

    Returns:
        qsystem (QSystem*): QSystem
    */

    return _sim->create_qsystem(num_qubits);
}

void
Host::delete_qsystem(QSystem*& qsystem)
{
    /*
    Deletes a qsystem

    Args:
        qsystem (QSystem*): qsystem to delete

    Returns:
        /
    */

    _sim->delete_qsystem(qsystem);
}

void
Host::send_qubit(const uint32_t& host, Qubit* qubit)
{
    /*
    Sends a qubit to another host

    Args:
        host (uint32_t): host to send qubit to
        qubit (Qubit): qubit to send

    Returns:
        /
    */

    q_channels_out.at(host)->put(qubit, _time);
}

Qubit*
Host::receive_qubit(const uint32_t& host)
{
    /*
    Receives a qubit from a given host

    Args:
        host (uint32_t): host to receive from

    Returns:
        qubit (Qubit): received qubit
    */

    std::tuple<double, Qubit*> m = q_channels_in.at(host)->get();
    double recv_time = std::get<0>(m);
    Qubit* q = std::get<1>(m);
    _time = std::max(_time, recv_time);
    return q;
}

Qubit*
Host::receive_qubit_wait(const uint32_t& host, uint32_t timeout)
{
    std::tuple<double, Qubit*> m = q_channels_in.at(host)->get_wait(timeout);
    double recv_time = std::get<0>(m);
    Qubit* q = std::get<1>(m);
    _time = std::max(_time, recv_time);
    return q;

}

void
Host::send_classical(const uint32_t& host, std::string message)
{
    /*
    Sends a classical message to another host

    Args:
        host (uint32_t): host to send qubit to
        message (std::string): message to send

    Returns:
        /
    */

    c_channels_out.at(host)->put(message, _time);
}

std::string
Host::receive_classical(const uint32_t& host)
{
    /*
    Receives a classical message from a given host

    Args:
        host (uint32_t): host to receive from

    Returns:
        message (std::string): received message
    */

    std::tuple<double, std::string> m = c_channels_in.at(host)->get();
    double recv_time = std::get<0>(m);
    std::string message = std::get<1>(m);
    _time = std::max(_time, recv_time);
    return message;

}

std::string
Host::receive_classical_wait(const uint32_t& host, uint32_t timeout)
{
    std::tuple<double, std::string> m = c_channels_in.at(host)->get_wait(timeout);
    double recv_time = std::get<0>(m);
    std::string message = std::get<1>(m);
    _time = std::max(_time, recv_time);
    return message;
}

void
Host::X(Qubit* q)
{
    /*
    Applys a X gate to the specified qubit and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to

    Returns:
        /
    */

    _time += single_gate_time;
    q->X();
}

void
Host::Y(Qubit* q)
{
    /*
    Applys a Y gate to the specified qubit and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to

    Returns:
        /
    */

    _time += single_gate_time;
    q->Y();
}

void
Host::Z(Qubit* q)
{
    /*
    Applys a Z gate to the specified qubit and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to

    Returns:
        /
    */

    _time += single_gate_time;
    q->Z();
}

void
Host::H(Qubit* q)
{
    /*
    Applys a H gate to the specified qubit and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to

    Returns:
        /
    */

    _time += single_gate_time;
    q->H();
}

void
Host::SX(Qubit* q)
{
    /*
    Applys a square root X gate to the specified qubit and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to

    Returns:
        /
    */

    _time += single_gate_time;
    q->SX();
}

void
Host::SY(Qubit* q)
{
    /*
    Applys a square root Y gate to the specified qubit and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to

    Returns:
        /
    */

    _time += single_gate_time;
    q->SY();
}

void
Host::SZ(Qubit* q)
{
    /*
    Applys a square root Z gate to the specified qubit and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to

    Returns:
        /
    */

    _time += single_gate_time;
    q->SZ();
}

void
Host::K(Qubit* q)
{
    /*
    Applys a K gate to the specified qubit and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to

    Returns:
        /
    */

    _time += single_gate_time;
    q->K();
}

void
Host::T(Qubit* q)
{
    /*
    Applys a T gate to the specified qubit and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to

    Returns:
        /
    */

    _time += single_gate_time;
    q->T();
}

void
Host::iSX(Qubit* q)
{
    /*
    Applys a square inverse root X gate to the specified qubit and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to

    Returns:
        /
    */

    _time += single_gate_time;
    q->iSX();
}

void
Host::iSY(Qubit* q)
{
    /*
    Applys a square inverse root Y gate to the specified qubit and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to

    Returns:
        /
    */

    _time += single_gate_time;
    q->iSY();
}

void
Host::iSZ(Qubit* q)
{
    /*
    Applys a square inverse root Z gate to the specified qubit and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to

    Returns:
        /
    */

    _time += single_gate_time;
    q->iSZ();
}

void
Host::iT(Qubit* q)
{
    /*
    Applys a square inverse T gate to the specified qubit and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to

    Returns:
        /
    */

    _time += single_gate_time;
    q->iT();
}

void
Host::iK(Qubit* q)
{
    /*
    Applys a square inverse K gate to the specified qubit and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to

    Returns:
        /
    */

    _time += single_gate_time;
    q->iK();
}

void
Host::Rx(Qubit* q, double& theta)
{
    /*
    Applys a Rx gate to the specified qubit with the given angle theta and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to
        theta (double): angle to rotate state about

    Returns:
        /
    */

    _time += single_gate_time;
    q->Rx(theta);
}

void
Host::Ry(Qubit* q, double& theta)
{
    /*
    Applys a Ry gate to the specified qubit with the given angle theta and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to
        theta (double): angle to rotate state about

    Returns:
        /
    */

    _time += single_gate_time;
    q->Ry(theta);
}

void
Host::Rz(Qubit* q, double& theta)
{
    /*
    Applys a Rz gate to the specified qubit with the given angle theta and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to
        theta (double): angle to rotate state about

    Returns:
        /
    */

    _time += single_gate_time;
    q->Rz(theta);
}

void
Host::PHASE(Qubit* q, double& theta)
{
    /*
    Applys a PHASE gate to the specified qubit with the given angle theta and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to
        theta (double): angle to rotate state about

    Returns:
        /
    */

    _time += single_gate_time;
    q->PHASE(theta);
}

void
Host::custom_gate(Qubit* q, matrix_c_d& gate)
{
    /*
    Applys a custom gate to the specified qubit with the given gate and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to
        gate (matrix_c_d): gate to apply to qubit

    Returns:
        /
    */

    _time += single_gate_time;
    q->custom_gate(gate);
}

void
Host::CNOT(Qubit* q, Qubit* target)
{
    /*
    Applys a CNOT gate to the specified qubits and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to
        target (Qubit): target qubit

    Returns:
        /
    */

    _time += double_gate_time;
    q->CNOT(target);
}

void
Host::CY(Qubit* q, Qubit* target)
{
    /*
    Applys a CY gate to the specified qubits and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to
        target (Qubit): target qubit

    Returns:
        /
    */

    _time += double_gate_time;
    q->CY(target);
}

void
Host::CZ(Qubit* q, Qubit* target)
{
    /*
    Applys a CZ gate to the specified qubits and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to
        target (Qubit): target qubit

    Returns:
        /
    */

    _time += double_gate_time;
    q->CZ(target);
}

void
Host::CH(Qubit* q, Qubit* target)
{
    /*
    Applys a CH gate to the specified qubits and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to
        target (Qubit): target qubit

    Returns:
        /
    */

    _time += double_gate_time;
    q->CH(target);
}

void
Host::CPHASE(Qubit* q, Qubit* target, double& theta)
{
    /*
    Applys a CPHASE gate to the specified qubits and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to
        target (Qubit): target qubit
        theta (double): angle theta to rotate state

    Returns:
        /
    */

    _time += double_gate_time;
    q->CPHASE(target, theta);
}

void
Host::CU(Qubit* q, Qubit* target, matrix_c_d& gate)
{
    /*
    Applys a CU gate to the specified qubits and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to
        target (Qubit): target qubit
        gate (matrix_c_d): angle theta to rotate state

    Returns:
        /
    */

    _time += double_gate_time;
    q->CU(target, gate);
}

void
Host::SWAP(Qubit* q, Qubit* target)
{
    /*
    Applys a SWAP gate to the specified qubits and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to
        target (Qubit): target qubit

    Returns:
        /
    */

    _time += 3 * double_gate_time;
    q->SWAP(target);
}

void
Host::TOFFOLI(Qubit* q, Qubit* control, Qubit* target)
{
    /*
    Applys a TOFFOLI gate to the specified qubits and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to
        control (Qubit): second control qubit
        target (Qubit): target qubit

    Returns:
        /
    */

    _time += triple_gate_time;
    q->TOFFOLI(control, target);
}

void
Host::CCU(Qubit* q, Qubit* control, Qubit* target, matrix_c_d& gate_s)
{
    /*
    Applys a CCU gate to the specified qubits and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to
        control (Qubit): second control qubit
        target (Qubit): target qubit
        gate_s (matrix_c_d): gate to apply

    Returns:
        /
    */

    _time += triple_gate_time;
    q->CCU(control, target, gate_s);
}

void
Host::CSWAP(Qubit* q, Qubit* target_1, Qubit* target_2)
{
    /*
    Applys a CCU gate to the specified qubits and updates host time with the respective time

    Args:
        q (Qubit): qubit to apply gate to
        target_1 (Qubit): first target qubit
        target_2 (Qubit): second target qubit

    Returns:
        /
    */

    _time += triple_gate_time;
    q->CSWAP(target_1, target_2);
}

void
Host::bell_state(Qubit* q, Qubit* target, const uint32_t& state)
{
    /*
    Changes the state of the qubits into one of the specified Bell states and updates the host time accordingly

    Args:
        q (Qubit): qubit to apply gate to
        target (Qubit): target qubit
        state (uint32_t): int between 0 and 3 indicating the bell state

    Returns:
        /
    */

    _time += single_gate_time + double_gate_time;

    if (state > 0)
    {
        _time += single_gate_time;
    }
    else if(state > 2)
    {
        _time += single_gate_time;
    }

    q->bell_state(target, state);
}

uint32_t
Host::measure(Qubit* q)
{
    /*
    Measures the specified qubit and updates the host time accordingly

    Args:
        q (Qubit): qubit to apply gate to

    Returns:
        res (uint32_t): measurement result
    */

    _time += measure_time;
    return q->measure();
}

uint32_t
Host::bsm(Qubit* q, Qubit* target)
{
    /*
    Performs a Bell state measurement on the specified qubits and updates the host time accordingly

    Args:
        q (Qubit): qubit to apply gate to
        target (Qubit): target qubit

    Returns:
        res (uint32_t): measurement result
    */

    _time += bsm_time;
    return q->bsm(target);
}

void
Host::join()
{
    /*
    Finishes this host execution

    Args:
        /

    Returns:
        /
    */

    _thread.join();
}

void
Host::start()
{
    /*
    Starts the host in a new thread, each derived host should implement this function

    Args:
        /

    Returns:
        /
    */
}

Simulation::Simulation()
{
    /*
    Initialises a simulation

    Args:
        /

    Returns:
        /
    */
}

Simulation::~Simulation()
{
    /*
    Destructs a Simulation

    Args:
        /

    Returns:
        /
    */
}

QSystem*
Simulation::create_qsystem(const uint32_t& num_qubits)
{
    /*
    Creates a new thread-safe QSystem with a given number of qubits

    Args:
        num_qubits (uint32_t): number of qubits in the QSystem

    Returns:
        qsys (QSystem*): pointer to the QSystem
    */

    return new QSystem(num_qubits);
}

void
Simulation::delete_qsystem(QSystem* qsystem)
{
    /*
    Deletes the given QSystem

    Args:
        qsystem (QSystem*): qsystem to delete

    Returns:
        /
    */

    for (Qubit* qubit: qsystem->_qubits)
        delete qubit;

    delete qsystem;
}

void
Simulation::combine_states(std::vector<QSystem*> qsys_l)
{
    /*
    Combines the states of the given QSystems into the first QSystem

    Args:
        qsys_l (vector<QSystem*>): vector of QSystems to combine

    Returns:
        /
    */

    uint32_t num_qubits_n = 0;
    std::vector<Qubit*> qubits_n;
    std::vector<matrix_c_d> states;
    for (QSystem* qsys : qsys_l)
        num_qubits_n += qsys->_num_qubits;

    qubits_n.reserve(num_qubits_n);
    states.reserve(num_qubits_n);

    for (QSystem* qsys : qsys_l)
    {
        states.emplace_back(qsys->_state);
        for (Qubit* q : qsys->_qubits)
            qubits_n.emplace_back(std::move(q));
    }

    QSystem* qsys_n = new QSystem(1);
    qsys_n->_num_qubits = num_qubits_n;
    qsys_n->_qubits = qubits_n;
    qsys_n->_state = tensor_operator(states);

    for (uint32_t i = 0; i < num_qubits_n; i++)
    {
        qsys_n->_qubits.at(i)->_index = i;
        qsys_n->_qubits.at(i)->_qsystem = qsys_n;
    }

    for (QSystem* qsys: qsys_l)
        delete qsys;
}

void
Simulation::ptrace(Qubit* qubit)
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

    qubit->_qsystem->_qubits.erase(qubit->_qsystem->_qubits.begin() + idx);

    for (uint32_t i = idx; i < qubit->_qsystem->_num_qubits; i++)
        qubit->_qsystem->_qubits.at(i)->_index = i;

    delete qubit;
}

void
Simulation::partial_trace(std::vector<Qubit*> qubit_l)
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

bool
Simulation::check_list(std::vector<QSystem*>& req)
{
    for(QSystem* el : req)
        for(std::vector<QSystem*> qsys : _qsys_l)
            for(QSystem* q : qsys)
                if(q == el)
                    return true;
    return false;
}

void
Simulation::update_requests()
{
    _qsys_l = std::vector<std::vector<QSystem*>>(_qubits_l.size(), {0});
    for(std::vector<Qubit*> q_l : _qubits_l)
    {
        std::vector<QSystem*> qsys_ids(q_l.size(), 0);
        for(Qubit* q : q_l)
        {
            qsys_ids.push_back(q->_qsystem);
        }
        _qsys_l.push_back(qsys_ids);
    }
}

void
Simulation::add_request(Host* host, std::vector<Qubit*> q_l)
{
    std::vector<QSystem*> qsys_ids(q_l.size());
    for(Qubit* q : q_l)
        qsys_ids.push_back(q->_qsystem);

    std::lock_guard<std::mutex> lock(_qsys_l_mutex);
    _qubits_l.push_back(q_l);
    _qsys_l.push_back(qsys_ids);
    _host_l.push_back(host);
    if(_qsys_l.size() == 1)
        host->_request_waiting = true;
    if(check_list(qsys_ids))
    {
        return;
    }
    else
    {
        host->_request_waiting = true;
    }
    
}

void
Simulation::complete_request(Host* host)
{
    std::lock_guard<std::mutex> lock(_qsys_l_mutex);
    host->_request_waiting = false;
    _host_l.erase(_host_l.begin());
    _qubits_l.erase(_qubits_l.begin());
    _qsys_l.erase(_qsys_l.begin());
    update_requests();
    if(_qsys_l.size())
        _host_l[0]->_request_waiting = true;
}

void
Simulation::add_host(Host* host)
{
    /*
    Adds a host to the simulation

    Args:
        host (Host*): pointer to the host

    Returns:
        /
    */

    _hosts.push_back(host);
}

void
Simulation::add_hosts(std::vector<Host*> hosts)
{
    /*
    Adds multiple hosts to the simulation

    Args:
        hosts (std::vector<Host*>): collection of hosts to add

    Returns:
        /
    */

    _hosts.reserve(_hosts.size() + distance(hosts.begin(), hosts.end()));
    _hosts.insert(_hosts.end(), hosts.begin(), hosts.end());
}

void
Simulation::run()
{
    /*
    Starts each of the hosts in the simulation and waits for the finishing of their threads

    Args:
        /

    Returns:
        /
    */

    for (Host* host : _hosts)
        host->start();

    for (Host* host : _hosts)
        host->join();
}

void
Simulation::run_with_infinity(std::vector<Host*> hosts, std::vector<Host*> routers)
{
    for (Host* host : _hosts)
        host->start();

    std::vector<bool> finished_hosts(hosts.size(), false);

    uint32_t sum = 0;
    while(sum != hosts.size())
    {
        for(uint32_t i = 0; i < hosts.size(); i++)
        {
            if(_hosts[i]->_stop_event && !finished_hosts[i])
            {
                finished_hosts[i] = true;
                sum += 1;
            }
        }       
    }

    std::cout << "Simulation: all hosts finished" << std::endl;

    for(auto router : routers)
    {
        router->_stop_event = true;
        router->join();
    }

}