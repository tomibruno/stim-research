import stim
import numpy
import scipy
import shutil
import matplotlib as mpl


def shors_code(rounds, error_model, noise):
    circuit = stim.Circuit()
    qubits = range(17)
    data = qubits[::2]
    measure = qubits[1::2]
    bit_flip_ancillas = [1, 3, 7, 9, 13, 15]
    phase_flip_ancillas = [5, 11]

    #circuit.append_operation(error_model, data, noise)
    circuit.append_operation("X", 0)
    #-----add z stabilizers-----
    for q in bit_flip_ancillas:
        circuit.append_operation("CNOT", [q-1, q])
        #circuit.append_operation("DEPOLARIZE2", [q-1, q], noise)
        circuit.append_operation("CNOT", [q+1, q])
        #circuit.append_operation("DEPOLARIZE2", [q+1, q], noise)
    
    circuit.append_operation("H", phase_flip_ancillas)

    #-----add x stabilizers-----
    for d in data[0:6]:
        circuit.append_operation("CZ", [d, 5])
        #circuit.append_operation("DEPOLARIZE2", [d, 5], noise)

    for d in data[3:9]:
        circuit.append_operation("CZ", [d, 11])
        #circuit.append_operation("DEPOLARIZE2", [d, 11], noise)
    
    #-----measure x and z stabilizers----
    #circuit.append_operation(error_model, bit_flip_ancillas, noise)
    circuit.append_operation("MR", bit_flip_ancillas)
    #circuit.append_operation(error_model, phase_flip_ancillas, noise)
    circuit.append_operation("MRX", phase_flip_ancillas)

    #-----reset phase flip ancillas-----
    circuit.append_operation("H", phase_flip_ancillas)
    
    for m in range(len(measure)):
        circuit.append_operation("DETECTOR", [stim.target_rec(-1-m), stim.target_rec(-1-m-8)])

    full_circuit = stim.Circuit()
    full_circuit.append_operation("M", bit_flip_ancillas)
    full_circuit.append_operation("M", phase_flip_ancillas)
    full_circuit += circuit * rounds
    return full_circuit

def shot(rounds, circuit):
    sample = circuit.compile_sampler().sample(1)[0]
    print("Regular Stabilizer Measurement:")
    for i in range(rounds):
        print("Round", i, ":", "".join("_1"[int(e)] for e in sample[(i+1)*8: (i+1)*8 + 8]))

def detector_shot(rounds, circuit):
    sample = circuit.compile_detector_sampler().sample(1)[0]
    print("Detector Stabilizer Measurement:")
    for i in range(rounds - 1):
        print("Round", i, "to", i+1, ":", "".join("_1"[int(e)] for e in sample[(i+1)*8: (i+1)*8 + 8]))

rounds = 10
noise = 0.05
circuit = shors_code(rounds, "DEPOLARIZE1", noise)

# write the svg diagram to a file
with open("circuit_diagram.svg", "w") as svg_file:
    svg_file.write(str(circuit.diagram('timeline-svg')))


shot(rounds, circuit)
print()
detector_shot(rounds, circuit)
