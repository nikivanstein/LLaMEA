import numpy as np

class EnhancedHybridPSODE(HybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.local_search_prob = 0.1
        self.base_local_search_radius = 0.1
        self.dynamic_radius_factor = 0.5

    def __call__(self, func):
        for t in range(self.budget):
            if np.random.rand() < self.local_search_prob:  # Local Search
                for i in range(self.population_size):
                    dynamic_radius = self.base_local_search_radius * (1 + self.dynamic_radius_factor * (func(self.particles[i]) - func(self.global_best)))
                    perturbation = np.random.uniform(-dynamic_radius, dynamic_radius, self.dim)
                    candidate = np.clip(self.particles[i] + perturbation, -5.0, 5.0)
                    if func(candidate) < func(self.particles[i]):
                        self.particles[i] = candidate.copy()

            else:  # Original HybridPSODE update
                super().__call__(func)
        return self.global_best