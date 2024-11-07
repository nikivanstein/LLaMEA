import numpy as np

class AdaptiveLocalSearchImprovedHybridPSODE(ImprovedHybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.local_search_prob = 0.1
        self.local_search_radius = 0.1
        
    def adaptive_local_search(self, func, candidate, current_best):
        func_diff = func(candidate) - func(current_best)
        if func_diff < 0:
            self.local_search_radius *= 1.1  # Increase radius for promising moves
        else:
            self.local_search_radius *= 0.9  # Decrease radius for non-improving moves
        
    def __call__(self, func):
        for t in range(self.budget):
            if np.random.rand() < self.local_search_prob:  # Local Search
                for i in range(self.population_size):
                    perturbation = np.random.uniform(-self.local_search_radius, self.local_search_radius, self.dim)
                    candidate = np.clip(self.particles[i] + perturbation, -5.0, 5.0)
                    if func(candidate) < func(self.particles[i]):
                        self.particles[i] = candidate.copy()
                        self.adaptive_local_search(func, candidate, self.particles[i])
            
            else:  # Original HybridPSODE update
                super().__call__(func)
        return self.global_best