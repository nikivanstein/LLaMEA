import numpy as np

class EnhancedHybridPSODE(HybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_prob = 0.2
        self.mutation_factor = 0.5
        
    def __call__(self, func):
        for t in range(self.budget):
            if np.random.rand() < self.mutation_prob:  # Mutation
                for i in range(self.population_size):
                    idxs = np.random.choice(np.arange(self.population_size), 3, replace=False)
                    mutant = self.particles[idxs[0]] + self.mutation_factor * (self.particles[idxs[1]] - self.particles[idxs[2]])
                    candidate = np.clip(mutant, -5.0, 5.0)
                    if func(candidate) < func(self.particles[i]):
                        self.particles[i] = candidate.copy()
            
            else:  # Original HybridPSODE update
                super().__call__(func)
        return self.global_best