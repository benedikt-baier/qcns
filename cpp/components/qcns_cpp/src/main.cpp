#include <iostream>
#include "qcns_cpp/linalg/matrices.h"
#include "qcns_cpp/linalg/matrix_utils.h"
#include "qcns_cpp/linalg/operator_utils.h"
#include "qcns_cpp/qubits/Qubit.h"
#include "qcns_cpp/qubits/QSystem.h"

#include <chrono>

using namespace qubits;

int main() {
    // Example usage of linear_algebra module
    QSystem q_sys = QSystem(5);
    Qubit qubit_1 = Qubit(&q_sys, 5);

    auto start = std::chrono::high_resolution_clock::now();
    qubit_1.X();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time for real matrix addition: " << elapsed.count() << " seconds\n";

    return 0;
}