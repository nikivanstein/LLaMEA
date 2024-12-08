import numpy as np

class EnhancedHybridPSODE(HybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.dynamic_perturbation_prob = 0.1
    
    def __call__(self, func):
        for t in range(self.budget):
            if np.random.rand() < self.dynamic_perturbation_prob:  # Dynamic Perturbation
                for i in range(self.population_size):
                    perturbation = np.random.normal(0, 1) * np.abs(self.global_best - self.particles[i])
                    candidate = np.clip(self.particles[i] + perturbation, -5.0, 5.0)
                    if func(candidate) < func(self.particles[i]):
                        self.particles[i] = candidate.copy()
            
            else:  # Original HybridPSODE update
                super().__call__(func)
        return self.global_best