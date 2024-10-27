import numpy as np
import random

class HybridParticleSwarmEvolutionary:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.particles = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_particle = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = func(self.particles[:, 0])
            self.best_particle = self.particles[np.argmin(self.particles[:, 0]), :]
            self.best_fitness = fitness

            # Evolutionary Strategy
            self.particles[np.random.choice(self.population_size, size=10, replace=False), :] = self.particles[np.random.choice(self.population_size, size=10, replace=False), :] + \
                                                                                      self.particles[np.random.choice(self.population_size, size=10, replace=False), :] * \
                                                                                      np.random.uniform(-0.1, 0.1, size=(10, self.dim))

            # Particle Swarm Optimization
            for _ in range(10):
                new_particle = np.random.uniform(-5.0, 5.0, self.dim)
                new_fitness = func(new_particle)
                if new_fitness < self.best_fitness:
                    self.best_particle = new_particle
                    self.best_fitness = new_fitness
                    self.particles[np.argmin(self.particles[:, 0]), :] = new_particle

            # Selection
            self.particles = self.particles[np.argsort(self.particles[:, 0])]
            self.population_size = self.population_size // 2

            # Mutation
            self.particles[np.random.choice(self.population_size, size=self.population_size, replace=False), :] += np.random.uniform(-0.1, 0.1, size=(self.population_size, self.dim))

            # Refine the strategy
            for i in range(self.population_size):
                if np.random.rand() < 0.25:
                    self.particles[i, :] += np.random.uniform(-0.05, 0.05, size=self.dim)

            # Check if the best particle is improved
            if self.best_fitness < func(self.best_particle):
                self.particles[np.argmin(self.particles[:, 0]), :] = self.best_particle

        return self.best_particle, self.best_fitness

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

hybrid_PSO_E = HybridParticleSwarmEvolutionary(budget=100, dim=2)
best_candidate, best_fitness = hybrid_PSO_E(func)
print(f"Best candidate: {best_candidate}, Best fitness: {best_fitness}")