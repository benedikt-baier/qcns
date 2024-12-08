#include "qcns_cpp/qubits/QSystem.h"
#include "qcns_cpp/linalg/matrices.h"
#include "qcns_cpp/linalg/operator_utils.h"

namespace qubits {
    QSystem::QSystem(const uint32_t& num_qubits) {
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
            _state = *linalg::_P0;
        }
        else
        {
            std::vector<std::string> keys(_num_qubits, "_P0");
            _state = linalg::tensor_operator(keys);
        }
    }

    QSystem::~QSystem() {
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

    Qubit* QSystem::qubit(const uint32_t& index) {
        /*
        Accesses a single qubit in the system given an index

        Args:
            index (uint32_t): index of the qubit

        Returns:
            qubit (Qubit): qubit with the given index
        */

        return _qubits[index];
    }

}