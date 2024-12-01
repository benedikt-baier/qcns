#include "../cpp/components/qubit.h"
#include <chrono>

int main() {
    QSystem one_sys = QSystem(10);
    Qubit qubit_1 = Qubit(&one_sys, 0);
    auto start = std::chrono::high_resolution_clock::now();
    qubit_1.X();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Elapsed time: " << duration.count() << " seconds" << std::endl;
    return 0;
}