import numpy as np

class ImprovedHybridPSODE(HybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.local_search_radius = 0.1
        self.diversity_threshold = 0.2
        self.max_local_search_prob = 0.5
        self.min_local_search_prob = 0.05
        self.mutation_factor = 0.5

    def __call__(self, func):
        for t in range(self.budget):
            diversity = np.std(self.particles)
            self.local_search_prob = max(self.min_local_search_prob, min(self.max_local_search_prob, diversity / self.diversity_threshold))

            if np.random.rand() < self.local_search_prob:  # Dynamic Local Search with Mutation
                for i in range(self.population_size):
                    perturbation = np.random.uniform(-self.local_search_radius, self.local_search_radius, self.dim)
                    mutation_vector = np.mean(np.array([np.random.normal(0, self.mutation_factor) * (self.global_best - self.particles[i]) for _ in range(self.dim)]), axis=0)
                    candidate = np.clip(self.particles[i] + perturbation + mutation_vector, -5.0, 5.0)
                    if func(candidate) < func(self.particles[i]):
                        self.particles[i] = candidate.copy()
            else:  # Original HybridPSODE update
                super().__call__(func)
        return self.global_best