import stim
import numpy
import scipy
import shutil
import pymatching

def encode(circuit):
  circuit.append_operation("CNOT", [0, 3])
  circuit.append_operation("CNOT", [0, 6])
  circuit.append_operation("H", 0)
  circuit.append_operation("H", 3)
  circuit.append_operation("H", 6)
  circuit.append_operation("CNOT", [0, 1])
  circuit.append_operation("CNOT", [0, 2])
  circuit.append_operation("CNOT", [3, 4])
  circuit.append_operation("CNOT", [3, 5])
  circuit.append_operation("CNOT", [6, 7])
  circuit.append_operation("CNOT", [6, 8])
  return circuit

  
def shor_code(rounds, error_model, noise):
  circuit = stim.Circuit()
  qubits = range(17)
  data = qubits[:9]
  measure = qubits[9:17]

  
  #--- encode logical qubit -----
  circuit = encode(circuit)

  #-----apply noise----
  if error_model == "CCN":
    circuit.append_operation("DEPOLARIZE1", data, noise)
  elif error_model == "custom":
    circuit.append_operation("Y", 5)

  for i in range(rounds):
    if error_model == "phen":
      circuit.append_operation("DEPOLARIZE1", data, noise)
    #-----set up stabilizers-----
    #Bit-flip stabilizers:
    #Z1
    circuit.append_operation("CX", [0, 9])
    circuit.append_operation("CX", [1, 9])

    #Z2
    circuit.append_operation("CX", [1, 10])
    circuit.append_operation("CX", [2, 10])

    #Z3
    circuit.append_operation("CX", [3, 11])
    circuit.append_operation("CX", [4, 11])
    
    #Z4
    circuit.append_operation("CX", [4, 12])
    circuit.append_operation("CX", [5, 12])

    #Z5
    circuit.append_operation("CX", [6, 13])
    circuit.append_operation("CX", [7, 13])
    
    #Z6
    circuit.append_operation("CX", [7, 14])
    circuit.append_operation("CX", [8, 14])

    circuit.append_operation("H", data)
    #Phase-flip stabilizers:
    for d in data[0:6]:
      circuit.append_operation("CX", [d, 15]) #X1
      circuit.append_operation("CX", [d+3, 16]) #X2
    circuit.append_operation("H", data)

    #add phenomenological error
    if error_model == "phen":
      circuit.append_operation("DEPOLARIZE1", measure, noise)

    circuit.append_operation("MR", measure)

    #add detectors to detect measurement errors
    if error_model == "phen":
      for m in range(len(measure)):
        circuit.append_operation("DETECTOR", [stim.target_rec(-1-m), stim.target_rec(-1-m-8)])
  
  
  full_circuit = stim.Circuit()

  # add dummy measurements to avoid comparing measurements before start of time
  if error_model == "phen": 
    full_circuit.append_operation("M", measure)

  full_circuit += circuit


  return full_circuit

def shot(circuit, rounds):
    sample = circuit.compile_sampler().sample(1)[0]
    
    for i in range(rounds):
      print("Round", i, ":", "".join("_1"[int(e)] for e in sample[i*8: i*8 + 8]))

      num_bit_flips = sum(sample[i*8:i*8+6])
      num_phase_flips = sum(sample[i*8+7:i*8+8])

      print("Most probable error:")
      if num_bit_flips == 0 and num_phase_flips == 0:
        print("No error!")
      elif num_bit_flips <= 2 and num_phase_flips <= 2:
        if sample[i*8] and sample[i*8+1]: print("Bit-flip error on qubit #1")
        elif sample[i*8]: print("Bit flip error on qubit #0")
        elif sample[i*8+1]: print("Bit flip error on qubit #2")
        elif sample[i*8+2] and sample[i*8+3]: print("Bit-flip error on qubit #4")
        elif sample[i*8+2]: print("Bit flip error on qubit #3")
        elif sample[i*8+3]: print("Bit flip error on qubit #5")
        elif sample[i*8+4] and sample[i*8+5]: print("Bit-flip error on qubit #7")
        elif sample[i*8+4]: print("Bit flip error on qubit #6")
        elif sample[i*8+5]: print("Bit flip error on qubit #8")

        if sample[i*8+6] and sample[i*8+7]: print("Phase-flip on L1 qubit #1")
        elif sample[i*8+6]: print("Phase-flip error on L1 qubit #0")
        elif sample[i*8+7]: print("Phase-flip error on L1 qubit #2")
      else:
        print("Logical error, cannot decode")
      print()
    
def detector_shot(circuit, rounds):
    sample = circuit.compile_detector_sampler().sample(1)[0]
    
    for i in range(rounds - 1):
      print("Round " + str(i) + " to " + str(i+1) + ": ", "".join("_1"[int(e)] for e in sample[i*8: i*8 + 8]))

rounds = 10

#circuit = shor_code(1, "CCN", 0.1)
#shot(circuit, 1)

circuit = shor_code(rounds, "phen", 0.02)
detector_shot(circuit, rounds)


with open("circuit_diagram.svg", "w") as svg_file:
    svg_file.write(str(circuit.diagram('timeline-svg')))
