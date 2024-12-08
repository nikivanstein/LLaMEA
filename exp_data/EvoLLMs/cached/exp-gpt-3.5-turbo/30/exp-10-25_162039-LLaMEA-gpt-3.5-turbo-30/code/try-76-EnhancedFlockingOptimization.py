# import numpy as np

class EnhancedFlockingOptimization(FlockingOptimization):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def _update_position(self, population, best_position, w_min=0.4, w_max=0.9, c1=0.8, c2=0.9):
        w = w_min + (w_max - w_min) * (self.budget - _) / self.budget  # Dynamic inertia weight
        r1, r2 = np.random.rand(self.budget, self.dim), np.random.rand(self.budget, self.dim)
        velocity = w * population + c1 * r1 * (best_position - population) + c2 * r2 * (population - best_position)
        return np.clip(population + velocity, -5.0, 5.0)