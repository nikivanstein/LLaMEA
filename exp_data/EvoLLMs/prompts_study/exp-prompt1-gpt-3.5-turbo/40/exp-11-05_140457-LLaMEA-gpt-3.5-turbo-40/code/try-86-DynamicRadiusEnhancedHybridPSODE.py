import numpy as np

class DynamicRadiusEnhancedHybridPSODE(EnhancedHybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.success_rate_threshold = 0.3
        self.max_radius = 1.0
        self.min_radius = 0.05

    def __call__(self, func):
        for t in range(self.budget):
            diversity = np.std(self.particles)
            self.local_search_prob = max(self.min_local_search_prob, min(self.max_local_search_prob, diversity / self.diversity_threshold))

            if np.random.rand() < self.local_search_prob:  # Dynamic Local Search
                for i in range(self.population_size):
                    perturbation = np.random.uniform(-self.local_search_radius, self.local_search_radius, self.dim)
                    candidate = np.clip(self.particles[i] + perturbation, -5.0, 5.0)
                    if func(candidate) < func(self.particles[i]):
                        self.particles[i] = candidate.copy()
                        self.local_search_radius = min(self.max_radius, self.local_search_radius * 1.1)  # Increase radius on successful perturbation
                    else:
                        self.local_search_radius = max(self.min_radius, self.local_search_radius * 0.9)  # Decrease radius on unsuccessful perturbation
            else:  # Original HybridPSODE update
                super().__call__(func)
        return self.global_best