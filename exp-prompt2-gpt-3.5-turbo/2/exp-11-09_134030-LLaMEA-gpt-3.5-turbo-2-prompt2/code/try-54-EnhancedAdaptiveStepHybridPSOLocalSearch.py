import numpy as np

class EnhancedAdaptiveStepHybridPSOLocalSearch(HybridPSOLocalSearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.memory = np.zeros(dim)  # Initialize memory for each dimension
        self.neighborhood_size = 3
        self.population_size = 10  # Introduce dynamic population size

    def local_search(self, particle, func):
        best_particle = np.copy(particle)
        step_size = 0.1
        for _ in range(5):
            for _ in range(self.neighborhood_size):
                new_particles = [np.clip(best_particle + step_size * np.random.randn(self.dim) + 0.2 * self.memory, self.lb, self.ub) for _ in range(self.population_size)]
                current_val = func(best_particle)
                new_vals = [func(new_p) for new_p in new_particles]
                best_idx = np.argmin(new_vals)
                if new_vals[best_idx] < current_val:
                    best_particle = np.copy(new_particles[best_idx])
                    step_size *= 0.9
                    self.memory = 0.9 * self.memory + 0.1 * (best_particle - particle)
                else:
                    step_size *= 1.1
            best_particle += 0.1 * np.random.randn(self.dim)
        return best_particle