import numpy as np

class AdaptiveHybridPSODE(ImprovedHybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        
    def __call__(self, func):
        for t in range(self.budget):
            if np.random.rand() < self.local_search_prob:  # Local Search
                for i in range(self.population_size):
                    perturbation = np.random.uniform(-self.local_search_radius / (np.log10(t + 2)), self.local_search_radius / (np.log10(t + 2)), self.dim)
                    candidate = np.clip(self.particles[i] + perturbation, -5.0, 5.0)
                    if func(candidate) < func(self.particles[i]):
                        self.particles[i] = candidate.copy()
            
            else:  # Original HybridPSODE update
                super().__call__(func)
        return self.global_best