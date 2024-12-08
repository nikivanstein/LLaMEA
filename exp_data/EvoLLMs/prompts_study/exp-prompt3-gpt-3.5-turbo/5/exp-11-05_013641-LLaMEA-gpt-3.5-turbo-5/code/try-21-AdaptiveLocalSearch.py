import numpy as np

class AdaptiveLocalSearch(HybridPSOLocalSearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.adaptive_step_size = 0.1

    def local_search(current_position):
        step_sizes = np.random.uniform(low=-self.adaptive_step_size, high=self.adaptive_step_size, size=self.dim)
        candidate_position = current_position + step_sizes
        return candidate_position