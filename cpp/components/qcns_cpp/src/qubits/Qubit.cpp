#include "qcns_cpp/qubits/Qubit.h"
#include "qcns_cpp/linalg/operator_utils.h"
#include "qcns_cpp/linalg/matrix_utils.h"
#include "qcns_cpp/linalg/matrices.h"

namespace qubits {
    
    Qubit::Qubit(QSystem* qsystem, uint32_t index) {
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
    
    Qubit::~Qubit() {
        /*
        Destructs a Qubit

        Args:
            /

        Returns:
            /
        */
    }

    void Qubit::X() {
        /*
        Applys a X gate to the qubit

        Args:
            /

        Returns:
            /
        */
        std::string key = "x_" + std::to_string(_qsystem->_num_qubits) + "_" + std::to_string(_index);
        Eigen::MatrixXcd gate = linalg::get_single_operator(key, *linalg::_X, _index, _qsystem->_num_qubits);
        linalg::multi_dot(gate, _qsystem->_state);
    }

}