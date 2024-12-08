import numpy as np

class EnhancedHybridPSODEImproved(HybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.local_search_radius = 0.1
        self.diversity_threshold = 0.2
        self.max_local_search_prob = 0.5
        self.min_local_search_prob = 0.05
        self.inertia_weight = 0.9  # New parameter for adaptive inertia weight

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
            else:  # Updated HybridPSODE with adaptive inertia weight
                for i in range(self.population_size):
                    cognitive = self.cognitive_weight * np.random.uniform(0, 1, self.dim) * (self.personal_best[i] - self.particles[i])
                    social = self.social_weight * np.random.uniform(0, 1, self.dim) * (self.global_best - self.particles[i])
                    velocity = self.inertia_weight * self.velocities[i] + cognitive + social
                    self.particles[i] = np.clip(self.particles[i] + velocity, -5.0, 5.0)
                    if func(self.particles[i]) < func(self.personal_best[i]):
                        self.personal_best[i] = self.particles[i].copy()
                    if func(self.particles[i]) < func(self.global_best):
                        self.global_best = self.particles[i].copy()
                    self.velocities[i] = velocity
                    self.inertia_weight -= 0.0005  # Adaptive update of inertia weight

        return self.global_best