import numpy as np

class DynamicLocalSearchImprovedHybridPSODE(ImprovedHybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        
    def __call__(self, func):
        for t in range(self.budget):
            dynamic_local_search_prob = 0.1 + 0.4 * (1 - t / self.budget)  # Dynamic probability
            dynamic_local_search_radius = 0.1 + 0.4 * (1 - t / self.budget)  # Dynamic radius
            
            for i in range(self.population_size):
                if np.random.rand() < dynamic_local_search_prob:
                    perturbation = np.random.uniform(-dynamic_local_search_radius, dynamic_local_search_radius, self.dim)
                    candidate = np.clip(self.particles[i] + perturbation, -5.0, 5.0)
                    if func(candidate) < func(self.particles[i]):
                        self.particles[i] = candidate.copy()
                else:
                    super().__call__(func)
        return self.global_best