import numpy as np

class AdaptedEnhancedHybridPSODE(EnhancedHybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.min_perturbation_scale = 0.1
        self.max_perturbation_scale = 0.5

    def __call__(self, func):
        for t in range(self.budget):
            diversity = np.std(self.particles)
            self.local_search_prob = max(self.min_local_search_prob, min(self.max_local_search_prob, diversity / self.diversity_threshold))

            if np.random.rand() < self.local_search_prob:  # Dynamic Local Search
                global_best_distance = np.linalg.norm(self.global_best)
                for i in range(self.population_size):
                    perturbation_scale = self.min_perturbation_scale + (self.max_perturbation_scale - self.min_perturbation_scale) * (np.linalg.norm(self.particles[i] - self.global_best) / global_best_distance)
                    perturbation = np.random.uniform(-perturbation_scale, perturbation_scale, self.dim)
                    candidate = np.clip(self.particles[i] + perturbation, -5.0, 5.0)
                    if func(candidate) < func(self.particles[i]):
                        self.particles[i] = candidate.copy()
            else:  # Original HybridPSODE update
                super().__call__(func)
        return self.global_best