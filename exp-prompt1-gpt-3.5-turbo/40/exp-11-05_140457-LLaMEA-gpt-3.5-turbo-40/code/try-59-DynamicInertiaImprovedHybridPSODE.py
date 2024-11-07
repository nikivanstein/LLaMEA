import numpy as np
from scipy.stats import levy
from numpy.random import default_rng

class DynamicInertiaImprovedHybridPSODE(ImprovedHybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.inertia_weight = 0.5
        self.inertia_decay = 0.99

    def __call__(self, func):
        rng = default_rng()
        for t in range(self.budget):
            self.inertia_weight *= self.inertia_decay
            for i in range(self.population_size):
                perturbation = np.random.uniform(-self.local_search_radius, self.local_search_radius, self.dim)
                velocity = self.inertia_weight * self.velocities[i] + self.c1 * rng.uniform(0, 1, self.dim) * (self.pbests[i] - self.particles[i]) + self.c2 * rng.uniform(0, 1, self.dim) * (self.global_best - self.particles[i])
                candidate = np.clip(self.particles[i] + velocity + perturbation, -5.0, 5.0)
                if func(candidate) < func(self.particles[i]):
                    self.particles[i] = candidate.copy()
        return self.global_best