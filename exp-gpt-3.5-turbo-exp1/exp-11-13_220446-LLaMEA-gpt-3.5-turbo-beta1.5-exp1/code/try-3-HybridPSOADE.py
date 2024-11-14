import numpy as np

class HybridPSOADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.particles = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        self.best_positions = self.particles.copy()
        self.best_fitness = np.full(self.population_size, np.inf)
    
    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.population_size):
                fitness = func(self.particles[i])
                if fitness < self.best_fitness[i]:
                    self.best_fitness[i] = fitness
                    self.best_positions[i] = self.particles[i].copy()
            best_global_index = np.argmin(self.best_fitness)
            best_global_position = self.best_positions[best_global_index]
            for i in range(self.population_size):
                r1, r2 = np.random.uniform(0, 1, 2)
                self.velocities[i] = 0.5 * self.velocities[i] + 1.5 * r1 * (self.best_positions[i] - self.particles[i]) + 1.5 * r2 * (best_global_position - self.particles[i])
                new_position = self.particles[i] + self.velocities[i]
                new_position = np.clip(new_position, -5.0, 5.0)
                if func(new_position) < self.best_fitness[i]:
                    self.particles[i] = new_position
                    self.best_fitness[i] = func(new_position)