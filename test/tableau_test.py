import stim
import numpy as np
from typing import List


def matrix_to_stabilizers(matrix: np.ndarray) -> List[stim.PauliString]:
    num_rows, num_cols = matrix.shape
    assert num_cols % 2 == 0
    num_qubits = num_cols // 2

    matrix = matrix.astype(np.bool_) #converts matrix entries to bool for pauli string construction
    return [
        stim.PauliString.from_numpy(
            xs=matrix[row, :num_qubits],
            zs=matrix[row, num_qubits:],
        )
        for row in range(num_rows)
    ]

def matrix_to_encoder(matrix: np.ndarray) -> stim.Circuit:
    stabilizers = matrix_to_stabilizers(matrix)
    tableau = stim.Tableau.from_stabilizers(
        stabilizers,
        allow_underconstrained=True,
    )
    return tableau.to_circuit(method='elimination')

def matrix_to_tableau(matrix: np.ndarray) -> stim.Tableau:
    stabilizers = matrix_to_stabilizers(matrix)
    tableau = stim.Tableau.from_stabilizers(
        stabilizers,
        allow_underconstrained=True,
    )
    return tableau

shor_code_matrix = np.array([
    [1,1,1,1,1,1,0,0,0, 0,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0, 1,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0, 0,1,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0, 0,0,0,1,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0, 0,0,0,0,1,1,0,0,0],
    [0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,1,1,0],
    [0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,1,1]
])

steane_code_matrix = np.array([
    [1,1,1,1,0,0,0, 0,0,0,0,0,0,0],
    [0,1,1,0,1,1,0, 0,0,0,0,0,0,0],
    [0,0,1,1,0,1,1, 0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0, 1,1,1,1,0,0,0],
    [0,0,0,0,0,0,0, 0,1,1,0,1,1,0],
    [0,0,0,0,0,0,0, 0,0,1,1,0,1,1],
])



def add_noise(circuit, qubits, noise) -> stim.Circuit:
    circuit.append("DEPOLARIZE1", qubits, noise)
    return circuit

# def add_measurement(matrix: np.ndarray, circuit: stim.Circuit) -> stim.Circuit:
#    stabilizers = matrix_to_stabilizers(matrix)
#    for s in stabilizers:
#       stabilizer_str = ""
#       for i in s.pauli_indices():
#          stabilizer_str.join(str(i) + "*")
#       #stabilizer_str = stabilizer_str[:-1]
#       #print(stabilizer_str)
#       print(s.pauli_indices())

def add_measurement_shor(circuit: stim.Circuit) -> stim.Circuit:
  circuit.append_from_stim_program_text('''
      MPP Z0*Z1
      MPP Z1*Z2
      MPP Z3*Z4
      MPP Z4*Z5
      MPP Z6*Z7
      MPP Z7*Z8
      MPP X0*X1*X2*X3*X4*X5
      MPP X3*X4*X5*X6*X7*X8
  ''')
  return circuit

def add_measurement_steane(circuit: stim.Circuit) -> stim.Circuit:
  circuit.append_from_stim_program_text('''
      MPP X0*X1*X2*X3
      MPP X1*X2*X4*X5
      MPP X2*X3*X5*X6
      MPP Z0*Z1*Z2*Z3
      MPP Z1*Z2*Z4*Z5
      MPP Z2*Z3*Z5*Z6
  ''')
  return circuit

shor_circuit = stim.Circuit()
shor_circuit = matrix_to_encoder(shor_code_matrix) + add_noise(shor_circuit, range(9), 0.1) + add_measurement_shor(shor_circuit)

steane_circuit = stim.Circuit()
steane_circuit = matrix_to_encoder(steane_code_matrix) + add_noise(steane_circuit, range(7), 0.1) + add_measurement_steane(steane_circuit)

shor_stabilizers = matrix_to_stabilizers(shor_code_matrix)

steane_stabilizers = matrix_to_stabilizers(steane_code_matrix)

c = stim.TableauSimulator()
#c.do_tableau(matrix_to_tableau(shor_code_matrix), range(9))
c.set_state_from_stabilizers(shor_stabilizers, allow_underconstrained=True)


c.x(0)

for s in shor_stabilizers:
   c.measure_observable(s)

print("".join(["_1"[int(i)] for i in c.current_measurement_record()]))

#add_measurement(shor_code_matrix, shor_circuit)



def shot(circuit: stim.Circuit):
    sample = circuit.compile_sampler().sample(1)[0]
    print("Stabilizer Measurement:")
    print("".join("_1"[int(e)] for e in sample))

#shot(steane_circuit)

with open("circuit_diagram.svg", "w") as svg_file:
    svg_file.write(str(steane_circuit.diagram('timeline-svg')))
