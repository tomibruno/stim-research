import stim

def shors_code(rounds, noise):
  circuit = stim.Circuit()
  
  circuit.append_operation("CNOT", [0, 3])
  circuit.append_operation("CNOT", [0, 6])
  
  pfdq = [0,3,6] # these are the qubits that will detect phase-flips
  circuit.append_operation("H", pfdq)

  for q in pfdq:
    circuit.append_operation("CNOT", [q, q+1])
    circuit.append_operation("CNOT", [q, q+2])
  
  

  