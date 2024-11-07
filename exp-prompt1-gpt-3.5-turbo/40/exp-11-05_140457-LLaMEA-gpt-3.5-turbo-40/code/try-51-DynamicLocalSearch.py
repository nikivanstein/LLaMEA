import numpy as np

class DynamicLocalSearch(ImprovedHybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.local_search_prob = 0.1
        self.min_local_search_radius = 0.01
        self.max_local_search_radius = 0.1
        
    def __call__(self, func):
        for t in range(self.budget):
            if np.random.rand() < self.local_search_prob:  # Local Search with dynamic radius
                diversity = np.std(self.particles, axis=0)
                dynamic_radius = np.clip(self.local_search_radius * (np.mean(diversity) / (np.max(diversity) + 1e-6)), self.min_local_search_radius, self.max_local_search_radius)
                for i in range(self.population_size):
                    perturbation = np.random.uniform(-dynamic_radius, dynamic_radius, self.dim)
                    candidate = np.clip(self.particles[i] + perturbation, -5.0, 5.0)
                    if func(candidate) < func(self.particles[i]):
                        self.particles[i] = candidate.copy()
            
            else:  # Original HybridPSODE update
                super().__call__(func)
        return self.global_best