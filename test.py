from qiskit import QuantumCircuit 
from qiskit_aer import AerSimulator 
qc = QuantumCircuit(2, 2) 
qc.h(0)          # superposition on qubit 0 
qc.cx(0, 1)      # entangle qubit 1 with qubit 0 
qc.measure([0, 1], [0, 1]) 
print(qc.draw()) 

result = AerSimulator().run(qc, shots=1000).result() 
print(result.get_counts())
