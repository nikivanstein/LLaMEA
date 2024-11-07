import numpy as np

class EnhancedHybridPSODE(HybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.local_search_prob = 0.1
        self.base_local_search_radius = 0.1
        self.mutation_factor = 0.01
        
    def __call__(self, func):
        for t in range(self.budget):
            if np.random.rand() < self.local_search_prob:  # Local Search
                for i in range(self.population_size):
                    perturbation = np.random.uniform(-self.local_search_radius, self.local_search_radius, self.dim)
                    candidate = np.clip(self.particles[i] + perturbation, -5.0, 5.0)
                    if func(candidate) < func(self.particles[i]):
                        self.particles[i] = candidate.copy()
                    else:  # Dynamic mutation strategy
                        self.local_search_radius *= np.exp(-self.mutation_factor*t/self.budget)
            
            else:  # Original HybridPSODE update
                super().__call__(func)
        return self.global_best