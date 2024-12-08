import numpy as np

class DynamicLocalSearch(ImprovedHybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.initial_local_search_prob = 0.1
        
    def __call__(self, func):
        diversity_threshold = 0.5 * self.population_size
        for t in range(self.budget):
            current_diversity = len(set(map(tuple, self.particles)))
            self.local_search_prob = self.initial_local_search_prob * (current_diversity / diversity_threshold)
            for i in range(self.population_size):
                if np.random.rand() < self.local_search_prob:  # Local Search
                    perturbation = np.random.uniform(-self.local_search_radius, self.local_search_radius, self.dim)
                    candidate = np.clip(self.particles[i] + perturbation, -5.0, 5.0)
                    if func(candidate) < func(self.particles[i]):
                        self.particles[i] = candidate.copy()
                else:  # Original HybridPSODE update
                    super().__call__(func)
        return self.global_best