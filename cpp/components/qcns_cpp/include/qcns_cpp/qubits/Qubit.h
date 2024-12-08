#pragma once

#include "qcns_cpp/qubits/QSystem.h"

#include <Eigen/Dense>


namespace qubits {
    class Qubit {
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

            void X();
            // void Y();
            // void Z();
            // void H();
            // void SX();
            // void SY();
            // void SZ();
            // void K();
            // void T();
            // void iSX();
            // void iSY();
            // void iSZ();
            // void iT();
            // void iK();
            // void Rx(double& theta);
            // void Ry(double& theta);
            // void Rz(double& theta);
            // void PHASE(double& theta);
            // void custom_gate(matrix_c_d& gate);
            // void CNOT(Qubit* target);
            // void CY(Qubit* target);
            // void CZ(Qubit* target);
            // void CH(Qubit* target);
            // void CPHASE(Qubit* target, double& theta);
            // void CU(Qubit* target, matrix_c_d& gate);
            // void SWAP(Qubit* target);
            // void TOFFOLI(Qubit* control, Qubit* target);
            // void CCU(Qubit* control, Qubit* target, matrix_c_d& gate_s);
            // void CSWAP(Qubit* target_1, Qubit* target_2);
            // void bell_state(Qubit* target, const uint32_t& state);

            // uint32_t measure(uint32_t basis=0);
            // uint32_t bsm(Qubit* target);

            // double fidelity(matrix_c_d& state);            
    };
}
