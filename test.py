import stim
import numpy
import scipy

c = stim.Circuit("""
                H 0
                CNOT 0 1
                M 0 1
                """)

def rep_code(distance, rounds, noise):
  circuit = stim.Circuit()
  # need n data qubits and n+1 measure qubits, alternating
  qubits = range(2*distance + 1) 
  data = qubits[::2]
  measure = qubits[1::2]

  circuit.append_operation("X_ERROR", data, noise)
  pairs1 = qubits[:-1]
  circuit.append_operation("CNOT", pairs1)
  pairs2 = qubits[1:][::-1]
  circuit.append_operation("CNOT", pairs2)
  circuit.append_operation("MR", measure)
  return circuit * rounds

def shot(circuit):
  sample = circuit.compile_sampler().sample(1)[0]
  print("".join(str(int(e)) for e in sample))



shot(rep_code(3, 2, 0.1))
