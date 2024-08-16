import stim
import matplotlib.pyplot as plt
import numpy as np
from typing import List
  
class GridNode:
   def __init__(self, valid, state):
      self.valid = valid
      self.state = state

class SurfaceCodeGrid:
    def __init__(self, distance):
      self.distance = distance  
      self.num_measurements = distance ** 2 - 1
      self.grid = [[GridNode(valid=True, state=False) for _ in range(distance+1)] for _ in range(distance+1)]
      self.map = {}
      i = 0
      for row in range(distance+1):
        for col in range(distance+1):
           if row == 0:
              if col == 0 or col % 2 == 1:
                 self.grid[row][col].valid = False
           elif row == distance:
              if col == distance or col % 2 == 0:
                 self.grid[row][col].valid = False
           elif col == 0:
              if row == distance or row % 2 == 0:
                 self.grid[row][col].valid = False
           elif col == distance:
              if row == 0 or row % 2 == 1:
                 self.grid[row][col].valid = False
           else:
              self.map[i] = [row, col]
              i+=1       
    
    def flip_measurement(self, index):
      if index in self.map:
        coords = self.map[index]
        m = self.grid[coords[0]][coords[1]].state
        self.grid[coords[0]][coords[1]].state = not m
       
    def get_measurement(self, index):
      if index in self.map:
        coords = self.map[index]
        return self.grid[coords[0]][coords[1]].state
    
    def print_grid(self):
      for row in range(self.distance + 1):
         line = ""
         for col in range(self.distance + 1):
            if not self.grid[row][col].valid:
               line += ' '
            else:
               line += "".join("_1"[int(self.grid[row][col].state)])
            line += ' '
         print(line)
    
    def clear_grid(self):
        self.grid = [[False for _ in range(self.num_measurements)] for _ in range(self.num_measurements)]

def print_raw(raw_sample, numShots, d):
  detector_rounds = d*3 - 1
  measurements_per_round = d**2 - 1
  
  for i in range(numShots):
    print("Shot", str(i) + ":")
    for r in range(detector_rounds):
      print("Round " + str(r) + " to " + str(r+1) + ":")
      for m in range(measurements_per_round):
        print(" ".join("_1"[int(raw_sample[i][r*measurements_per_round + m])]))
      print()
    print("====================")

def get_data_sample(raw_sample, numShots, d):
  detector_rounds = d*3 - 1
  measurements_per_round = d**2 - 1
  data_sample = [[False for _ in range(measurements_per_round)] for _ in range(numShots)]
  for shot in range(numShots):
    for r in range(detector_rounds):
      for m in range(measurements_per_round):
          if raw_sample[shot][r*measurements_per_round + m]:
            e = data_sample[shot][m]
            data_sample[shot][m] = not e
  return data_sample

def count_errors(numShots, noise, d):
  circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                rounds=d*3,
                distance=d,
                after_clifford_depolarization=noise,
                after_reset_flip_probability=noise,
                before_measure_flip_probability=noise,
                before_round_data_depolarization=noise,
            )
  
  sampler = circuit.compile_detector_sampler()

  raw_sample = sampler.sample(numShots)
  data_sample = get_data_sample(raw_sample, numShots, d)
  codes = [SurfaceCodeGrid(d) for _ in range(numShots)]

  print_raw(raw_sample, numShots, d)

  for shot in range(numShots):
     for i, e in enumerate(data_sample[shot]):
        if e:
           codes[shot].flip_measurement(i)
     
        
  
  
  numComplexErrors = 0
  numTrivialErrors = 0
  for i in range(numShots):
    codes[i].print_grid()
    hasComplex = hasComplexError(codes[i], d)
    if hasComplex:
       numComplexErrors += 1
    elif not hasComplex and hasError(codes[i], d):
       numTrivialErrors += 1
    else:
       print("No errors")
    print()
  return numComplexErrors, numTrivialErrors
  
def hasError(code, d):
  for row in range(d+1):
    for col in range(d+1):
      if code.grid[row][col].valid and code.grid[row][col].state:
        print("Trivial error at (" + str(row) + ", " + str(col) + ")")
        return True
  return False

def hasComplexError(code, d):
    for row in range(d+1):
       for col in range(d+1):
          a = code.grid[row][col]
          if a.valid and a.state:
             p = code.grid[row+1][col+1]
             q = code.grid[row-1][col+1]
             r = code.grid[row+1][col-1]
             s = code.grid[row-1][col-1]
             num_valid = sum(int(x.valid) for x in [p,q,r,s])
             if num_valid > 3 and hasEvenParity(p.state,q.state,r.state,s.state):
                print("Complex error at (" + str(row) + ", " + str(col) + ")")
                return True
    return False
      
def hasEvenParity(p, q, r, s):
   return (int(p)+int(q)+int(r)+int(s)) % 2 == 0          


def plot3(distances, noise_levels, numShots):
    colors = ['red', 'blue', 'green']
    labels = ['Complex', 'Trivial', 'None'] 
    
    fig, axs = plt.subplots(1, len(distances), figsize=(12, 4), sharey=True)
    
    bar_width = 0.8
    
    for i, d in enumerate(distances):
        plt.rcParams.update({'font.size':15})
        complex_percentages = []
        trivial_percentages = []
        none_percentages = []
        
        for noise in noise_levels[i]:
          errors = count_errors(numShots, noise, d)
          total_errors = sum(errors)
          complex_percentages.append(errors[0] / numShots * 100)
          trivial_percentages.append(errors[1] / numShots * 100)
          none_percentages.append((numShots - total_errors) / numShots * 100)
        ax = axs
        if len(distances) != 1:
           ax = axs[i]
        
        x_positions = range(len(noise_levels[i]))
        
        
        c = ax.bar(x_positions, complex_percentages, bottom = [triv + none for triv, none in zip(trivial_percentages, none_percentages)], width=bar_width, color=colors[0], label=labels[0])
        t = ax.bar(x_positions, trivial_percentages, bottom = none_percentages, width=bar_width, color=colors[1], label=labels[1])
        n = ax.bar(x_positions, none_percentages, width=bar_width, color=colors[2], label=labels[2])
        
        ax.set_xlabel(f"Physical Error Rate")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(["{:.2E}".format(n) for n in noise_levels[i]])
        
        ax.set_ylabel("Percentage")
        ax.set_title(f"Distance {d}")
        ax.legend()
    plt.tight_layout()
    plt.show()

error_rates = [0.0001, 0.0005, 0.001, 0.003, 0.005]
distances = [5,7,9]
noise_levels = [
   error_rates,
   [e / 5 for e in error_rates],
   [e / 10 for e in error_rates]
]



print(count_errors(10, 0.01, 5))
#plot3(distances, noise_levels, numShots=10000)
d=5
noise = 0.005
circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                rounds=d*3,
                distance=d,
                after_clifford_depolarization=noise,
                after_reset_flip_probability=noise,
                before_measure_flip_probability=noise,
                before_round_data_depolarization=noise,
            )



# Get a surface code circuit.
circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    distance=5,
    rounds=15,
    after_clifford_depolarization=1e-3,
    before_round_data_depolarization=1e-3,
    before_measure_flip_probability=1e-3,
    after_reset_flip_probability=1e-3,
)

# Truncate the circuit so it stops just before the data measurements.
last_measurement_layer = len(circuit) - 1
while circuit[last_measurement_layer].name != 'MR':
    last_measurement_layer -= 1
circuit = circuit[:last_measurement_layer]


# Collect stats over a million+ shots, in batches of 1024 (takes ~10 seconds).
i_hits = np.zeros(shape=circuit.num_qubits, dtype=np.uint64)
x_hits = np.zeros(shape=circuit.num_qubits, dtype=np.uint64)
y_hits = np.zeros(shape=circuit.num_qubits, dtype=np.uint64)
z_hits = np.zeros(shape=circuit.num_qubits, dtype=np.uint64)
for _ in range(1000):
    sim = stim.FlipSimulator(
        batch_size=1024,
        disable_stabilizer_randomization=True,
    )
    sim.do(circuit)

    # Count number of times each Pauli occurred on each qubit.
    instance_paulis: stim.PauliString
    for instance_paulis in sim.peek_pauli_flips():
        xs, zs = instance_paulis.to_numpy(bit_packed=False)
        i_hits += ~xs & ~zs
        x_hits += xs & ~zs
        y_hits += xs & zs
        z_hits += ~xs & zs


# Print results.
qubit_coords = circuit.get_final_qubit_coordinates()
for q, coords in qubit_coords.items():
    i, x, y, z = i_hits[q], x_hits[q], y_hits[q], z_hits[q]
    t = i + x + y + z
    print(f"qubit at {str(coords):>14}: X={x/t:<16} Y={y/t:<16} Z={z/t:<16}")

      
