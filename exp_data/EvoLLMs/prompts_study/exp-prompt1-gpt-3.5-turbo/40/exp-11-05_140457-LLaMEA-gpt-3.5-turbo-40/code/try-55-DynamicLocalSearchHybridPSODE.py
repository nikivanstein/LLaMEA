import numpy as np

class DynamicLocalSearchHybridPSODE(HybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.local_search_prob = 0.1
        self.min_local_search_radius = 0.05
        self.max_local_search_radius = 0.2
        
    def __call__(self, func):
        for t in range(self.budget):
            # Adapt local search radius based on performance
            current_radius = self.min_local_search_radius + (self.max_local_search_radius - self.min_local_search_radius) * (t / self.budget)
            
            if np.random.rand() < self.local_search_prob:  # Local Search
                for i in range(self.population_size):
                    perturbation = np.random.uniform(-current_radius, current_radius, self.dim)
                    candidate = np.clip(self.particles[i] + perturbation, -5.0, 5.0)
                    if func(candidate) < func(self.particles[i]):
                        self.particles[i] = candidate.copy()
            
            else:  # Original HybridPSODE update
                super().__call__(func)
        return self.global_best