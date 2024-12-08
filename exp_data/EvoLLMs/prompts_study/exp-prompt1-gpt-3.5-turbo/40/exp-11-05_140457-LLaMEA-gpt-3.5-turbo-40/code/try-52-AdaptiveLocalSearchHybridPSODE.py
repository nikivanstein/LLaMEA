import numpy as np

class AdaptiveLocalSearchHybridPSODE(ImprovedHybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.local_search_prob = 0.1

    def __call__(self, func):
        for t in range(self.budget):
            if np.random.rand() < self.local_search_prob:  # Local Search
                for i in range(self.population_size):
                    perturbation_radius = np.random.uniform(0.01, 0.2)  # Adaptive perturbation based on landscape
                    perturbation = np.random.uniform(-perturbation_radius, perturbation_radius, self.dim)
                    candidate = np.clip(self.particles[i] + perturbation, -5.0, 5.0)
                    if func(candidate) < func(self.particles[i]):
                        self.particles[i] = candidate.copy()
            else:  
                super().__call__(func)
        return self.global_best