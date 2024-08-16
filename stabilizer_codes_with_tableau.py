import stim
import numpy as np
from typing import List
import pymatching
import matplotlib.pyplot as plt


# converts stabilizers defined by binary symplectic matrix into list of PauliStrings
def matrix_to_stabilizers(matrix: np.ndarray) -> List[stim.PauliString]:
    num_rows, num_cols = matrix.shape
    num_qubits = num_cols // 2

    matrix = matrix.astype(np.bool_) #converts matrix entries to bool for pauli string construction
    return [
        stim.PauliString.from_numpy(
            xs=matrix[row, :num_qubits],
            zs=matrix[row, num_qubits:],
        )
        for row in range(num_rows)
    ]

def encode_with_noise(matrix: np.ndarray, noise: float) -> stim.Circuit:
    stabilizers = matrix_to_stabilizers(matrix)
    tableau = stim.Tableau.from_stabilizers(stabilizers, allow_underconstrained=True)
    encoded_circuit = tableau.to_circuit(method='elimination')
    encoded_circuit.to_file("encoding.txt")

    file = open("encoding.txt")
    noisy_circuit = stim.Circuit()
    lines = file.readlines()
    for line in lines:
      operation, qubits = line.split(' ', 1)
      qubits = [int(i) for i in qubits.split()]
      if line[0] == 'C':
        for i in range(0,len(qubits),2):
          noisy_circuit.append(operation, [qubits[i], qubits[i+1]])
          noisy_circuit.append("DEPOLARIZE2", [qubits[i], qubits[i+1]], noise)
      else:
        for i in range(len(qubits)):
          noisy_circuit.append(operation, qubits[i])
          noisy_circuit.append("DEPOLARIZE1", qubits[i], noise)

    with open("encoded_circuit.svg", "w") as svg_file:
      svg_file.write(str(encoded_circuit.diagram('timeline-svg')))

    return noisy_circuit

def measure_with_noise(circuit: stim.Circuit, matrix: np.ndarray, noise: float, rounds: int):
    num_rows, num_cols = matrix.shape
    num_qubits = num_cols // 2
    for _ in range(rounds):
      for row in range(num_rows):
        operation = "MPP(" + str(noise) + ") "
        for q in range(num_qubits):
          x = matrix[row][q]
          z = matrix[row][num_qubits + q]
          if x or z:
            if x and z: operation += "Y" + str(q)
            elif x: operation += "X" + str(q) 
            else: operation += "Z" + str(q)
            operation += "*"
        
        if operation[-1] == "*": operation = operation[:-1]
        circuit.append_from_stim_program_text(operation)
    
      for m in range(num_rows+1):
        circuit.append_operation("DETECTOR", [stim.target_rec(-1-m), stim.target_rec(-1-m-num_rows)])

    full_circuit = stim.Circuit()
    # add dummy measurements to avoid comparing measurements before start of time
    full_circuit.append_operation("M", range(num_rows+1))
    full_circuit += circuit

    obs_measurement = "MPP(" + str(noise) + ") "
    for i in range(circuit.num_qubits):
        obs_measurement += "Z" + str(i) + "*"
    obs_measurement = obs_measurement[:-1]
    full_circuit.append_from_stim_program_text(obs_measurement)
    full_circuit.append_from_stim_program_text('''OBSERVABLE_INCLUDE(0) rec[-1]''')
    
    return full_circuit

def sample(circuit: stim.Circuit, stabilizers: List[stim.PauliString], noise: float, rounds: int):
    sample = circuit.compile_sampler().sample(1)[0]
    print("Regular Stabilizer Measurement:")
    for r in range(rounds):
        print("Round", r,":", "".join(["_1"[int(i)] for i in sample[(r+1)*len(stabilizers) + 1: (r+2)*len(stabilizers)+ 1]]))
        for s_num, i in enumerate(sample[(r+1)*len(stabilizers): (r+2)*len(stabilizers)]):
            if int(i): print("   Stabilizer " + str(stabilizers[s_num]) + " anticommuted")
    
    print()
    
def detector_sample(circuit: stim.Circuit, stabilizers: List[stim.PauliString], noise: float, rounds: int):
    sample = circuit.compile_detector_sampler().sample(1)[0]
    print("Detector Stabilizer Measurement:")
    for r in range(rounds - 1):
        print("Round", r, "to", r+1, ":", "".join("_1"[int(e)] for e in sample[(r+1)*len(stabilizers): (r+2)*len(stabilizers)]))
        for s_num, i in enumerate(sample[(r+1)*len(stabilizers): (r+2)*len(stabilizers)]):
          if int(i): print("   Stabilizer " + str(stabilizers[s_num]) + " measured different results between round " + str(r) + " to " + str(r+1))

def count_logical_errors(circuit: stim.Circuit, num_shots: int) -> int:
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    dem = circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)

    predictions = matcher.decode_batch(detection_events)

    num_errors = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1

    return num_errors

def plot(num_shots):
    for i , matrix in enumerate([shor_code_matrix, steane_code_matrix, quantum_hamming_15_11_code]):
        xs = []
        ys = []
        for noise in [0.0001, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
            circuit = encode_with_noise(matrix, noise)
            circuit = measure_with_noise(circuit, matrix, noise, rounds)
            num_errors_sampled = count_logical_errors(circuit, num_shots)
            xs.append(noise)
            ys.append(num_errors_sampled / num_shots)
        if i==0: plt.plot(xs, ys, label="shor")
        elif i==1: plt.plot(xs, ys, label="steane")
        else: plt.plot(xs, ys, label="15,11 quantum hamming")
    plt.loglog()
    plt.xlabel("physical error rate")
    plt.ylabel("logical error rate per shot")
    plt.legend()
    plt.show()

shor_code_matrix = np.array([
  [0,0,0,0,0,0,0,0,0, 1,1,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0, 0,1,1,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,0,0, 0,0,0,1,1,0,0,0,0],
  [0,0,0,0,0,0,0,0,0, 0,0,0,0,1,1,0,0,0],
  [0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,1,1,0],
  [0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,1,1],
  [1,1,1,1,1,1,0,0,0, 0,0,0,0,0,0,0,0,0],
  [0,0,0,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0]
])




steane_code_matrix = np.array([
  [1,1,1,1,0,0,0, 0,0,0,0,0,0,0],
  [0,1,1,0,1,1,0, 0,0,0,0,0,0,0],
  [0,0,1,1,0,1,1, 0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0, 1,1,1,1,0,0,0],
  [0,0,0,0,0,0,0, 0,1,1,0,1,1,0],
  [0,0,0,0,0,0,0, 0,0,1,1,0,1,1]
])

quantum_hamming_15_11_code = np.array([
   [1,0,0,0,1,1,1,0,0,0,0,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,1,0,0,1,0,0,1,1,0,1,0,1,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,1,0,0,1,0,1,0,1,1,1,0,1,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,1,0,0,1,0,1,1,1,1,1,0,1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1,0,0,0,1,1,1,0,0,0,0,1,1,1,1],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,1,0,0,1,0,0,1,1,0,1,0,1,1,1],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,1,0,0,1,0,1,0,1,1,1,0,1,1],
   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,1,0,0,1,0,1,1,1,1,1,0,1]
])


matrix = shor_code_matrix
noise = 0
rounds = 4


circuit = encode_with_noise(matrix, noise)
circuit = measure_with_noise(circuit, matrix, noise, rounds)

#sample(circuit, matrix_to_stabilizers(matrix), noise, rounds)
#detector_sample(circuit, matrix_to_stabilizers(matrix), noise, rounds)


num_shots = 100000
# num_logical_errors = count_logical_errors(circuit, num_shots)
# print("Logical Error Rate:", str(round(num_logical_errors/num_shots * 100, 4)) + "%", "(there were", num_logical_errors, "logical errors out of", num_shots, "shots)")

plot(num_shots)

with open("circuit_diagram.svg", "w") as svg_file:
    svg_file.write(str(circuit.diagram('timeline-svg')))

