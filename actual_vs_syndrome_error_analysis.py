import stim
import numpy as np
import matplotlib.pyplot as plt


class Qubit:
  def __init__(self, valid):
    self.valid = valid
    self.xErr = False
    self.zErr = False
    self.state = False

class SurfaceCode:
  def __init__(self, d, circuit: stim.Circuit):
    self.distance = d
    self.grid = [[Qubit(False) for _ in range(2*d+1)] for _ in range(2*d+1)]
    
    self.circuit = circuit
    self.detector_rounds = 1
    
    # with open("sc_diagram.svg", "w") as svg_file:
    #   svg_file.write(str(self.circuit.diagram('timeline-svg')))

    self.measurements = self.circuit.compile_detector_sampler().sample(1)[0]
    self.det_map = self.circuit.get_detector_coordinates()
    # fill in grid with valid qubits
    qubit_coords = self.circuit.get_final_qubit_coordinates()
    for _, coords in qubit_coords.items():
      row = coords[1]
      col = coords[0]
      row = int(row)
      col = int(col)
      self.grid[row][col].valid = True
    
    # fill in stabilizer index to coord map
    self.stab_map = {}
    self.z_stab_map = {}
    i = 0
    z = 0
    for col in range(len(self.grid[0])):
       for row in range(len(self.grid)):
          if self.grid[row][col].valid and not self.is_data(row, col) and not self.is_x_stabilizer(row, col):
             self.z_stab_map[z] = [row, col]
             z += 1

    for row in range(len(self.grid)):
       for col in range(len(self.grid[row])):
          if self.grid[row][col].valid and not self.is_data(row, col):
             self.stab_map[i] = [row, col]
             i += 1
   
  def is_data(self, row, col):
     return row % 2 == 1 and col % 2 == 1
     
  def fill_with_actual_errors(self):
    
    # truncate the circuit so it stops just before the data measurements.
    # last_measurement_layer = len(self.circuit) - 1
    # while self.circuit[last_measurement_layer].name != 'MR':
    #     last_measurement_layer -= 1
    # self.circuit = self.circuit[:last_measurement_layer]
      
    fs = stim.FlipSimulator(batch_size=1, disable_stabilizer_randomization=True)
    fs.do(self.circuit)

    # fill grid with errors from flip simulator
    for pauli_errors in fs.peek_pauli_flips():
        xs, zs = pauli_errors.to_numpy(bit_packed=False)
        for q, coords in self.circuit.get_final_qubit_coordinates().items():
          row = int(coords[1])
          col = int(coords[0])


          qubit_type = "z-stabilizer"
          if self.is_data(row, col):
            qubit_type = "data"
          elif self.is_x_stabilizer(row, col):
            qubit_type = "x-stabilizer"
          
          if xs[q]:
            #print("X error on " + qubit_type + " qubit (" + str(coords[1]) + ", " + str(coords[0]) + ")")
            self.grid[row][col].xErr = not self.grid[row][col].xErr                  
          if zs[q]:
            #print("Z error on " + qubit_type + " qubit (" + str(coords[1]) + ", " + str(coords[0]) + ")")
            self.grid[row][col].zErr = not self.grid[row][col].zErr
          
  def is_x_stabilizer(self, row, col):
     if row % 4 == 0:
        return col % 4 == 2
     else:
        return col % 4 == 0

  def fill_with_stabilizer_measurements(self):

    measurements_per_round = self.distance**2 - 1
    stab_to_flip_at_end = set()
    # first fill z-stabilizer measurements in first round (x-stabilizer is non-deterministic),
    for m in range(int(measurements_per_round / 2)):
       if self.measurements[m]:
          coords = self.z_stab_map[m]
          row = int(coords[0])
          col = int(coords[1])
          #print("Round 1: flipping z stabilizer at (" + str(row) + ", " + str(col) + ")")
          self.grid[row][col].state = True
             
  
    # then fill from rest of rounds with both stabilizer measurements
    for r in range(self.detector_rounds - 1):
      for m in range(measurements_per_round):
        if self.measurements[int(measurements_per_round / 2) - 1 + r*measurements_per_round + m]:
          
          coords = self.stab_map[m]
          row = int(coords[0])
          col = int(coords[1])

          type = "z"
          if m not in self.z_stab_map:
             type = "x"

          #print("Round " + str(r+2) + ": flipping " + type + " stabilizer at (" + str(row) + ", " + str(col) + ")")
          # if self.grid[row][col].state:
          #    if m not in stab_to_flip_at_end:
          #       stab_to_flip_at_end.add(m)
          #    continue
          self.grid[row][col].state = not self.grid[row][col].state
    
      # print("Stabilizers after round", r)
      # self.print_stabilizer_measurements()

    # flip stabilizers that had two flips in detector record
    # for m in range(measurements_per_round):
    #    if m in stab_to_flip_at_end:
    #       coords = self.stab_map[m]
    #       self.grid[coords[0]][coords[1]].state = False
      
  def has_error_string(self, with_stabilizers) -> bool:
    if with_stabilizers:
      for row in range(len(self.grid)):
        for col in range(len(self.grid[row])):
          a = self.grid[row][col]
          if a.valid and not self.is_data(row, col) and a.state:
            if row-2<0 or col-2<0 or row+2>=len(self.grid) or col+2>=len(self.grid[row]):
               continue
            p = self.grid[row+2][col+2]
            q = self.grid[row-2][col+2]
            r = self.grid[row+2][col-2]
            s = self.grid[row-2][col-2]
            num_valid = sum(int(x.valid) for x in [p,q,r,s])
            if num_valid > 3 and self.hasEvenParity(p.state,q.state,r.state,s.state):
              return True
    else:
      # check actual errors
      directions = [[-2,0],[-2,2],[0,2],[2,2],[2,0],[2,-2],[0,-2],[-2,-2]]

      for i in range(len(self.grid)):
          for j in range(len(self.grid[i])):
            if self.is_data(i, j):
              if self.grid[i][j].xErr:
                for d in directions:
                   if i + d[0] < 0 or i + d[0] >= 2 * self.distance or j + d[1] < 0 or j + d[1] >= self.distance:
                      continue
                   if self.is_data(i+d[0], j+d[1]) and self.grid[i+d[0]][j+d[1]].xErr:
                      return True
                
              if self.grid[i][j].zErr:
                for d in directions:
                   if i + d[0] < 0 or i + d[0] >= 2 * self.distance or j + d[1] < 0 or j + d[1] >= self.distance:
                      continue
                   if self.grid[i+d[0]][j+d[1]].zErr:
                      return True 
    return False
  
  def hasEvenParity(self, p, q, r, s):
    return (int(p)+int(q)+int(r)+int(s)) % 2 == 0          

  def has_error(self, with_stabilizers) -> bool:
    if with_stabilizers:
      for row in range(len(self.grid)):
        for col in range(len(self.grid[row])):
          if self.grid[row][col].valid and self.grid[row][col].state:
            return True
    else:
      for i in range(len(self.grid)):
        for j in range(len(self.grid[i])):
          if self.is_data(i, j):
            if self.grid[i][j].xErr or self.grid[i][j].zErr:
              return True
    return False
          
  def print_grid_errors(self, only_data):
    idx = 0
    for row in range(len(self.grid)):
      line = ""
      for col in range(len(self.grid[row])):
          q = self.grid[row][col]
          if not q.valid or (only_data and not self.is_data(row, col)):
            line += "   "
          else:
            if q.xErr and q.zErr:
                line += "|Y|"
            elif q.xErr:
                line += "|X|"
            elif q.zErr:
                line += "|Z|"
            else:
                line += "|.|"
      print(line)

  def print_stabilizer_measurements(self):
    for row in range(len(self.grid)):
      line = ""
      for col in range(len(self.grid[row])):
        if not self.grid[row][col].valid:
          line += "   "
        elif self.is_data(row, col):
          if self.grid[row][col].xErr and self.grid[row][col].zErr:
             line += "|Y|"
          elif self.grid[row][col].xErr:
             line += "|X|"
          elif self.grid[row][col].zErr:
             line += "|Z|"
          else:
             line += "|.|"
        else:
          line += "|"
          line += "".join("_1"[int(self.grid[row][col].state)])
          line += "|"
      print(line)

  def clear_grid(self):
      for i in range(len(self.grid)):
        for j in range(len(self.grid[i])):
            self.grid[i][j].state = False


def countActualErrors(numShots, d, noise):
   num_none = 0
   num_triv = 0
   num_string = 0
   for i in range(numShots):
      circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                rounds=1,
                distance=d,
                after_clifford_depolarization=0,
                after_reset_flip_probability=0,
                before_measure_flip_probability=0,
                before_round_data_depolarization=noise,
            )
      sc = SurfaceCode(d, circuit)
    
      sc.fill_with_actual_errors()
      if sc.has_error_string(with_stabilizers=False):
         num_string += 1
      elif sc.has_error(with_stabilizers=False):
         num_triv += 1
      else:
         num_none += 1

   return num_none, num_triv, num_string
      
def countStabErrors(numShots, d, noise):
  num_none = 0
  num_triv = 0
  num_string = 0
  for i in range(numShots):
      circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                rounds=1,
                distance=d,
                after_clifford_depolarization=0,
                after_reset_flip_probability=0,
                before_measure_flip_probability=0,
                before_round_data_depolarization=noise,
            )
      sc = SurfaceCode(d, circuit)
      sc.fill_with_stabilizer_measurements()
      if sc.has_error_string(with_stabilizers=True):
         num_string += 1
      elif sc.has_error(with_stabilizers=True):
         num_triv += 1
      else:
         num_none += 1
      
  return num_none, num_triv, num_string
         
def plot(distances, noise_levels, numShots):
    colors = ['red', 'blue', 'green']
    labels = ['Complex', 'Trivial', 'None'] 
    
    fig, axs = plt.subplots(1, len(distances), figsize=(12, 4), sharey=True)
    
    bar_width = 0.8
    graph_idx = 0
    i = 0
    for d in distances:
        complex_percentages = []
        trivial_percentages = []
        none_percentages = []
        
        for noise in noise_levels:
          errors = countActualErrors(numShots, d, noise)
          complex_percentages.append(errors[2] / numShots * 100)
          trivial_percentages.append(errors[1] / numShots * 100)
          none_percentages.append(errors[0] / numShots * 100)
        
        real_ax = axs
        if axs is iter:
          real_ax = axs[i]
        
        x_positions = range(len(noise_levels))
        
        
        c = real_ax.bar(x_positions, complex_percentages, bottom = [triv + none for triv, none in zip(trivial_percentages, none_percentages)], width=bar_width, color=colors[0], label=labels[0])
        t = real_ax.bar(x_positions, trivial_percentages, bottom = none_percentages, width=bar_width, color=colors[1], label=labels[1])
        n = real_ax.bar(x_positions, none_percentages, width=bar_width, color=colors[2], label=labels[2])
        
        real_ax.set_xlabel(f"Physical Error Rate")
        real_ax.set_xticks(x_positions)
        real_ax.set_xticklabels(["{:.2E}".format(n) for n in noise_levels])
        
        real_ax.set_ylabel("Percentage")
        real_ax.set_title(f"Distance {d} (Actual)")
        
        # pred_ax = axs[i+1]
        # i+=2
        # c = pred_ax.bar(x_positions, complex_percentages_p, bottom = [triv + none for triv, none in zip(trivial_percentages_p, none_percentages_p)], width=bar_width, color=colors[0], label=labels[0])
        # t = pred_ax.bar(x_positions, trivial_percentages_p, bottom = none_percentages_p, width=bar_width, color=colors[1], label=labels[1])
        # n = pred_ax.bar(x_positions, none_percentages_p, width=bar_width, color=colors[2], label=labels[2])
        
        # pred_ax.set_xlabel(f"Physical Error Rate")
        # pred_ax.set_xticks(x_positions)
        # pred_ax.set_xticklabels(["{:.2E}".format(n) for n in noise_levels])
        
        # pred_ax.set_ylabel("Percentage")
        # pred_ax.set_title(f"Distance {d} (Predicted with Stabilizers)")
    if axs is iter:
       axs[-1].legend()
    else:
       axs.legend()
    plt.tight_layout()
    plt.show()

def insertCustomError(circuit, lineNum, data_qubit_index):
    circuit.to_file("surface_code.txt")
    file = open("surface_code.txt")
    new_circuit = stim.Circuit()
    lines = file.readlines()
    for i, line in enumerate(lines):
      new_circuit.append_from_stim_program_text(line)
      if i == lineNum:
         new_circuit.append("Z_ERROR", data_qubit_index, 1)

    return new_circuit

distances = [81]
noise_levels = [0.005]

plot(distances, noise_levels, numShots=10000)

d = 7
noise = 0.001
circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                rounds=1,
                distance=d,
                after_clifford_depolarization=0, # applies depolarizing noise after every gate
                after_reset_flip_probability=0, # applies bit-flip noise on every qubit after reset
                before_measure_flip_probability=0, # applies bit-flip noise on measurement qubits right before measurement
                before_round_data_depolarization=noise, # applies depolarizing noise on data qubits right before stabilizer measurement.
            )

newCircuit = insertCustomError(circuit, 108, 67)

# sc = SurfaceCode(d, newCircuit)
# sc.fill_with_actual_errors()
# print("Actual Errors:")
# sc.print_grid_errors(only_data=False)
# if sc.has_error_string(with_stabilizers=False):
#    print("Has Error String")
# elif sc.has_error(with_stabilizers=False):
#    print("Has isolated error")
# else:
#    print("no error")

with open("scdiagram.svg", "w") as svg_file:
      svg_file.write(str(newCircuit.diagram('timeline-svg')))

full_noise_circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                rounds=2,
                distance=d,
                after_clifford_depolarization=0.01,
                after_reset_flip_probability=0.02,
                before_measure_flip_probability=0.03,
                before_round_data_depolarization=0.04,
            )

with open("full_error_sc_diagram.svg", "w") as svg_file:
   svg_file.write(str(full_noise_circuit.diagram('timeline-svg')))


# print()
# sc.fill_with_stabilizer_measurements()
# print("Stabilizer Measurements:")
# sc.print_stabilizer_measurements()
# if sc.has_error_string(with_stabilizers=True):
#    print("Has Error String")
# elif sc.has_error(with_stabilizers=True):
#    print("Has isolated error")
# else:
#    print("no error")


