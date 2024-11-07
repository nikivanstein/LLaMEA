import numpy as np

class EnhancedHybridPSODE(HybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.local_search_prob = 0.1
        self.local_search_radius = 0.1
        self.mutation_prob = 0.2
        self.mutation_rate = 0.1

    def __call__(self, func):
        for t in range(self.budget):
            if np.random.rand() < self.local_search_prob:  # Local Search
                for i in range(self.population_size):
                    perturbation = np.random.uniform(-self.local_search_radius, self.local_search_radius, self.dim)
                    candidate = np.clip(self.particles[i] + perturbation, -5.0, 5.0)
                    if func(candidate) < func(self.particles[i]):
                        self.particles[i] = candidate.copy()
            
            else:  # Original HybridPSODE update
                super().__call__(func)
            
            if np.random.rand() < self.mutation_prob:  # Mutation
                indices_to_mutate = np.random.choice(self.population_size, int(self.mutation_rate*self.population_size), replace=False)
                for i in indices_to_mutate:
                    mutation = np.random.uniform(-self.mutation_rate, self.mutation_rate, self.dim)
                    self.particles[i] = np.clip(self.particles[i] + mutation, -5.0, 5.0)
        return self.global_best