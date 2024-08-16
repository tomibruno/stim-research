import stim
import numpy
import scipy
import shutil

def rep_code(distance, rounds, noise):
  circuit = stim.Circuit()
  qubits = range(2*distance + 1)
  data = qubits[::2]
  measure = qubits[1::2]

  pairs1 = qubits[:-1]
  pairs2 = qubits[1:][::-1]
  circuit.append_operation("CNOT", pairs1)
  circuit.append_operation("DEPOLARIZE2", pairs1, noise)
  circuit.append_operation("CNOT", pairs2)
  circuit.append_operation("DEPOLARIZE2", pairs2, noise)
  
  circuit.append_operation("DEPOLARIZE1", qubits, noise)
  circuit.append_operation("MR", measure)

  for m in range(len(measure)):
    circuit.append_operation("DETECTOR", [stim.target_rec(-1-m), stim.target_rec(-1-m-distance)])


  full_circuit = stim.Circuit()
  full_circuit.append_operation("M", measure)
  full_circuit += circuit * rounds

  full_circuit.append_operation("M", data)
  for k in range(len(measure)):
    full_circuit.append_operation("DETECTOR", [stim.target_rec(-1-k), stim.target_rec(-2-k), stim.target_rec(-2-k-distance)])
  full_circuit.append_operation("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
  return full_circuit

def shot(circuit):
  sample = circuit.compile_sampler().sample(1)[0]
  print("".join("_1"[e] for e in sample))

def detect_shot(circuit):
  sample = circuit.compile_detector_sampler().sample(1, append_observables=True)[0]
  print("".join("_1"[e] for e in sample))
  
circuit = rep_code(distance=shutil.get_terminal_size().columns, rounds=10, noise=0.01)
detect_shot(circuit)

