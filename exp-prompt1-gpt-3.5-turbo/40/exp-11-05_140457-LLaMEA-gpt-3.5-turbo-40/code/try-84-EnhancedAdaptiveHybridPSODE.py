import numpy as np

class EnhancedAdaptiveHybridPSODE(HybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.local_search_radius = 0.1
        self.diversity_threshold = 0.2
        self.max_local_search_prob = 0.5
        self.min_local_search_prob = 0.05
        self.perturbation_factor = 0.1

    def __call__(self, func):
        for t in range(self.budget):
            diversity = np.std(self.particles)
            self.local_search_prob = max(self.min_local_search_prob, min(self.max_local_search_prob, diversity / self.diversity_threshold))

            if np.random.rand() < self.local_search_prob:
                for i in range(self.population_size):
                    perturbation = np.random.uniform(-self.local_search_radius * self.perturbation_factor, self.local_search_radius * self.perturbation_factor, self.dim)
                    candidate = np.clip(self.particles[i] + perturbation, -5.0, 5.0)
                    if func(candidate) < func(self.particles[i]):
                        self.particles[i] = candidate.copy()
            else:
                super().__call__(func)
        return self.global_best