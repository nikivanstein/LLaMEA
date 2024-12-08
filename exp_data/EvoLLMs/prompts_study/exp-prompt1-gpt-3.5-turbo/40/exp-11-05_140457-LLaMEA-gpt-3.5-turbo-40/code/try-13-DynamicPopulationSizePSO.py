import numpy as np

class DynamicPopulationSizePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.max_velocity = 0.5
        self.c1 = 2.0
        self.c2 = 2.0
        self.population_size = 20
        self.particles = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        self.global_best = None
        self.global_best_fitness = float('inf')
    
    def __call__(self, func):
        for t in range(self.budget):
            # Dynamic population size adaptation based on fitness diversity
            self.population_size = int(5 + 15 * np.exp(-0.5*t/self.budget))
            self.particles = np.vstack([self.particles, np.random.uniform(-5.0, 5.0, (self.population_size-self.particles.shape[0], self.dim))])
            self.velocities = np.vstack([self.velocities, np.zeros((self.population_size-self.velocities.shape[0], self.dim))])

            for i in range(self.population_size):
                fitness = func(self.particles[i])
                if fitness < self.global_best_fitness:
                    self.global_best = self.particles[i].copy()
                    self.global_best_fitness = fitness
                pbest = self.particles[i].copy()
                if fitness < func(pbest):
                    pbest = self.particles[i].copy()
                inertia_weight = 0.5 + 0.4 * np.exp(-0.5*t/self.budget)
                cognitive_component = self.c1 * np.random.rand(self.dim) * (pbest - self.particles[i])
                social_component = self.c2 * np.random.rand(self.dim) * (self.global_best - self.particles[i])
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_component + social_component
                self.velocities[i] = np.clip(self.velocities[i], -self.max_velocity, self.max_velocity)
                self.particles[i] = np.clip(self.particles[i] + self.velocities[i], -5.0, 5.0)
                
            # Diversity Maintenance
            centroid = np.mean(self.particles[:self.population_size], axis=0)
            for i in range(self.population_size):
                self.particles[i] += 0.03 * (centroid - self.particles[i])

        return self.global_best