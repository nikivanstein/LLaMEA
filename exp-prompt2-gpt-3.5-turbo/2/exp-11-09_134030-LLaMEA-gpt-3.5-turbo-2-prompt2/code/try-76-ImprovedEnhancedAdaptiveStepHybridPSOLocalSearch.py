import numpy as np

class ImprovedEnhancedAdaptiveStepHybridPSOLocalSearch(EnhancedAdaptiveStepHybridPSOLocalSearch):
    def local_search(self, particle, func):
        best_particle = np.copy(particle)
        step_size = 0.1
        mutation_rate = 0.1
        for _ in range(5):
            for _ in range(self.neighborhood_size):
                new_particle = np.clip(best_particle + step_size * np.random.randn(self.dim) + 0.2 * self.memory, self.lb, self.ub)
                current_val = func(best_particle)
                new_val = func(new_particle)
                if new_val < current_val:
                    best_particle = np.copy(new_particle)
                    step_size *= 0.9
                    self.memory = 0.9 * self.memory + 0.1 * (best_particle - particle)
                    mutation_rate = 0.1 + 0.1 * np.linalg.norm(best_particle - particle)  # Dynamic mutation rate based on fitness
                else:
                    step_size *= 1.1
            best_particle += mutation_rate * np.random.randn(self.dim)
        return best_particle