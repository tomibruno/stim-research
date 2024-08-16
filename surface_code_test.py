import stim
class GridNode:
   def __init__(self, valid, state):
      self.valid = valid
      self.state = state

class SurfaceCodeGrid:
    def __init__(self, distance):
      self.distance = distance  
      self.num_measurements = distance ** 2 - 1 # Number of measurements in each round
      self.grid = [[GridNode(True, False) for _ in range(distance+1)] for _ in range(distance+1)]
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
                     
    def is_valid(self, index):
       coords = self.map[index]
    
    def flip_measurement(self, index):
       coords = self.map[index]
       m = self.grid[coords[0]][coords[1]].measurement
       self.grid[coords[0]][coords[1]].measurement = not m
       
    
    def get_measurement(self, index):
      coords = self.map[index]
      return self.grid[coords[0]][coords[1]].measurement
    
    def print_grid(self):
      for row in range(self.distance + 1):
         line = ""
         for col in range(self.distance + 1):
            if not self.grid[row][col].valid:
               line.append(' ')
            else:
               line.append(str(int(self.grid[row][col].valid)))
         print(line)
    
    def clear_grid(self):
        self.grid = [[False for _ in range(self.num_measurements)] for _ in range(self.num_measurements)]

# Example usage:
if __name__ == "__main__":
    distance = 3  # Example distance for the surface code
    surface_grid = SurfaceCodeGrid(distance)
    
    # Set some example measurements
    surface_grid.set_measurement(0, 0, True)
    surface_grid.set_measurement(1, 4, True)
    surface_grid.set_measurement(2, 8, True)
    
    # Visualize the grid
    surface_grid.visualize_grid()
    
    # Get a measurement
    print("Measurement at (1, 4):", surface_grid.get_measurement(1, 4))
    
    # Clear the grid
    surface_grid.clear_grid()
    
    # Visualize the cleared grid
    print("\nCleared Grid:")
    surface_grid.visualize_grid()
