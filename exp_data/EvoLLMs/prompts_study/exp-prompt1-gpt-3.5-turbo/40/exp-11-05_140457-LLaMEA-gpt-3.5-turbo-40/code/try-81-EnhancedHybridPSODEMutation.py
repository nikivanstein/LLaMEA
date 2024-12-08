import numpy as np

class EnhancedHybridPSODEMutation(EnhancedHybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_rate = 0.1

    def __call__(self, func):
        for t in range(self.budget):
            diversity = np.std(self.particles)
            self.local_search_prob = max(self.min_local_search_prob, min(self.max_local_search_prob, diversity / self.diversity_threshold))

            if np.random.rand() < self.local_search_prob:  # Dynamic Local Search with Mutation
                for i in range(self.population_size):
                    perturbation = np.random.uniform(-self.local_search_radius, self.local_search_radius, self.dim)
                    candidate = np.clip(self.particles[i] + perturbation, -5.0, 5.0)
                    if func(candidate) < func(self.particles[i]):
                        self.particles[i] = candidate.copy()
                    else:  # Mutation mechanism
                        mutation = np.random.uniform(-self.mutation_rate, self.mutation_rate, self.dim)
                        candidate = np.clip(self.particles[i] + mutation, -5.0, 5.0)
                        if func(candidate) < func(self.particles[i]):
                            self.particles[i] = candidate.copy()
            else:  # Original HybridPSODE update
                super().__call__(func)
        return self.global_best