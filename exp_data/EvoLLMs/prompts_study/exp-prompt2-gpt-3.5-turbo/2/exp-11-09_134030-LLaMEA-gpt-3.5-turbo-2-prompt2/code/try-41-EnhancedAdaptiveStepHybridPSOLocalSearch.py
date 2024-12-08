import numpy as np

class EnhancedAdaptiveStepHybridPSOLocalSearch(HybridPSOLocalSearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.memory = np.zeros(dim)  # Initialize memory for each dimension

    def local_search(self, particle, func):
        best_particle = np.copy(particle)
        step_size = 0.1
        for _ in range(5):
            new_particle = np.clip(best_particle + step_size * np.random.randn(self.dim) + 0.2 * self.memory, self.lb, self.ub)
            current_val = func(best_particle)
            new_val = func(new_particle)
            if new_val < current_val:
                best_particle = np.copy(new_particle)
                step_size *= 0.9  # Decrease step size if better solution found
                self.memory = 0.8 * self.memory + 0.2 * (best_particle - particle)  # Update memory dynamically
            else:
                step_size *= 1.1  # Increase step size if no improvement
        return best_particle