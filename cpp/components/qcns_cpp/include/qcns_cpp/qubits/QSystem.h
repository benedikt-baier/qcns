#pragma once

#include "qcns_cpp/qubits/Qubit.h"

#include <Eigen/Dense>
#include <vector>


namespace qubits {
    class Qubit;
    class QSystem;
    class QSystem {
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
            Eigen::MatrixXcd _state;

            QSystem(const uint32_t& _num_qubits = 1);
            ~QSystem();
            
            Qubit* qubit(const uint32_t& index);
    };
}