import stim
import numpy
import scipy
import shutil


def rep_code(distance, rounds, noise):
  circuit = stim.Circuit()
  # need n+1 data qubits and n measure qubits, alternating
  qubits = range(2*distance + 1) 
  data = qubits[::2]
  measure = qubits[1::2]

  # set up quantum circuit for repitition code
  pairs1 = qubits[:-1]
  circuit.append_operation("CNOT", pairs1)
  pairs2 = qubits[1:][::-1]
  circuit.append_operation("CNOT", pairs2)

  # add general noise to each of the gates
  circuit.append_operation("DEPOLARIZE2", pairs1, noise) # DEPOLARIZE2 used for noise on 2 qubit gates
  circuit.append_operation("DEPOLARIZE2", pairs2, noise)
  circuit.append_operation("DEPOLARIZE1", qubits, noise) 

  # measure and reset the measurement qubits
  circuit.append_operation("MR", measure)

  # detect whether sequential measurements differ for each measurement bit
  for k in range(len(measure)):
    circuit.append_operation("DETECTOR", [stim.target_rec(-1-k), stim.target_rec(-1-k-distance)]) 

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
  print("".join("_1"[int(e)] for e in sample))

def detect_shot(circuit):
  sample = circuit.compile_detector_sampler().sample(1)[0]
  print("".join("_1"[int(e)] for e in sample))

  

# shot(rep_code(shutil.get_terminal_size().columns, 20, 0.01))
# print("\n")
detect_shot(rep_code(3, 1, 0.1))
