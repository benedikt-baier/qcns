#include "../cpp/components/qubit.h"
// // #include "../cpp/components/qcns_cpp/include/qcns_cpp/qubit/QSystem.h"
// // #include "../cpp/components/qcns_cpp/include/qcns_cpp/qubit/Qubit.h"
// #include "qcns_cpp/qubit/Qubit.h"
// #include "qcns_cpp/qubit/QSystem.h"
#include <chrono>

// using namespace qubit;

void 
test_old_cpp(const uint32_t& num_qubits = 1)
{
    QSystem q_sys = QSystem(num_qubits);
    Qubit qubit_1 = Qubit(&q_sys, 5);

    auto start = std::chrono::high_resolution_clock::now();
    qubit_1.X();
    auto end = std::chrono::high_resolution_clock::now();
}

void
test_new_cpp(const uint32_t& num_qubits = 1)
{
    QSystem q_sys = QSystem(num_qubits);
    Qubit qubit_1 = Qubit(&q_sys, 5);

    auto start = std::chrono::high_resolution_clock::now();
    qubit_1.X();
    auto end = std::chrono::high_resolution_clock::now();
}

int main() {
    uint32_t num_qubits = 10;
    test_old_cpp(num_qubits);
    test_new_cpp(num_qubits);

    return 0;
}