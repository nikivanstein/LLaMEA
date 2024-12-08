import numpy as np

class EnhancedHybridPSODE(HybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.local_search_prob = 0.1
        self.local_search_radius = 0.1
        self.mutation_prob = 0.2
        self.mutation_scale = 0.5
        
    def __call__(self, func):
        for t in range(self.budget):
            if np.random.rand() < self.local_search_prob:  # Local Search
                for i in range(self.population_size):
                    perturbation = np.random.uniform(-self.local_search_radius, self.local_search_radius, self.dim)
                    candidate = np.clip(self.particles[i] + perturbation, -5.0, 5.0)
                    if func(candidate) < func(self.particles[i]):
                        self.particles[i] = candidate.copy()
            
            else:  # Original HybridPSODE update with mutation
                for i in range(self.population_size):
                    if np.random.rand() < self.mutation_prob:
                        best_index = np.argsort([func(p) for p in self.particles])[0]
                        mutation_vector = self.particles[best_index] - self.particles[i]
                        perturbed_particle = self.particles[i] + self.mutation_scale * mutation_vector
                        self.particles[i] = np.clip(perturbed_particle, -5.0, 5.0)
                super().__call__(func)
        
        return self.global_best