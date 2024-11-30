#include <random>
#include "host.h"

class Host_1 : public Host
{
public:

    using Host::Host;

    void run()
    {
        std::random_device rd;
        std::mt19937 g(rd());

        QSystem* qsys = _sim->create_qsystem(3);
        std::vector<Qubit> qubits = qsys->_qubits;

        std::uniform_real_distribution<double> dist(0.0, 3.141592);
        std::vector<size_t> arrangement = { 0, 1, 2 };

        std::shuffle(arrangement.begin(), arrangement.end(), g);

        std::vector<double> thetas(3);
        thetas[0] = dist(g);
        thetas[1] = dist(g);
        thetas[2] = dist(g);

        for (size_t i = 0; i < 3; i++)
        {
            if (arrangement[i] == 0)
                qubits[0].Rx(thetas[i]);
            if (arrangement[i] == 1)
                qubits[0].Ry(thetas[i]);
            if (arrangement[i] == 2)
                qubits[0].Rz(thetas[i]);
        }

        qubits[1].H();
        qubits[1].CNOT(qubits[2]);

        send_qubit(2, qubits[2]);

        qubits[0].CNOT(qubits[1]);
        qubits[0].H();

        size_t res = 2 * qubits[0].measure() + qubits[1].measure();

        send_classical(2, std::to_string(res));
        
    }

    void start()
    {
        _thread = std::thread(&Host_1::run, this);
    }
};

class Host_2 : public Host
{
public:

    using Host::Host;

    void run()
    {
        Qubit q = receive_qubit(1);
        std::string res = receive_classical(1);

        if (res == "1")
            q.X();
        if (res == "2")
            q.Z();
        if (res == "3")
            q.Y();

        _sim->delete_qsystem(q._qsystem);

    }

    void start()
    {
        _thread = std::thread(&Host_2::run, this);
    }
};

int
main()
{
    Simulation* sim = new Simulation();

    Host_1 host_1 = Host_1(1, sim);
    Host_2 host_2 = Host_2(2, sim);

    host_1.set_connection(host_2);

    sim->add_hosts({ &host_1 ,&host_2 });

    sim->run();

    delete sim;

    return 0;
}