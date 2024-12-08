import numpy as np
from scipy.stats import levy

class EnhancedDynamicMutationPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.max_velocity = 0.5
        self.c1 = 2.0
        self.c2 = 2.0
        self.particles = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        self.global_best = None
        self.global_best_fitness = float('inf')
    
    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.population_size):
                fitness = func(self.particles[i])
                if fitness < self.global_best_fitness:
                    self.global_best = self.particles[i].copy()
                    self.global_best_fitness = fitness
                pbest = self.particles[i].copy()
                if fitness < func(pbest):
                    pbest = self.particles[i].copy()
                inertia_weight = 0.9 - 0.5 * _ / self.budget
                cognitive_component = self.c1 * np.random.rand(self.dim) * (pbest - self.particles[i])
                social_component = self.c2 * np.random.rand(self.dim) * (self.global_best - self.particles[i])
                levy_flight = levy.rvs(size=self.dim)  # Levy flight for global exploration
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_component + social_component + levy_flight
                self.velocities[i] = np.clip(self.velocities[i], -self.max_velocity, self.max_velocity)
                self.particles[i] = np.clip(self.particles[i] + self.velocities[i], -5.0, 5.0)
        return self.global_best