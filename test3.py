import stim
class SurfaceCode:
  def __init__(self, d, noise):
    self.circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                rounds=d*3,
                distance=d,
                after_clifford_depolarization=noise,
                after_reset_flip_probability=noise,
                before_measure_flip_probability=noise,
                before_round_data_depolarization=noise,
            )
    
sc = SurfaceCode(d=5, noise=0.01)
print(sc.circuit.compile_detector_sampler().sample(1))
