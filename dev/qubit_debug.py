import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../qcns/python/components')))

from qubit import Qubit
from qubit import QSystem

num_qubits = 2

qsys = QSystem(num_qubits, fidelity=1, sparse=0)
qbit_0 = Qubit(qsys,0)
qbit_1 = Qubit(qsys,1)
print(f"Default system: \n{qsys}")
qbit_0.X()
print(f"Invert first Qubit: \n{qsys}")
qbit_1.X()
print(f"Invert second Qubit: \n{qsys}")
pass