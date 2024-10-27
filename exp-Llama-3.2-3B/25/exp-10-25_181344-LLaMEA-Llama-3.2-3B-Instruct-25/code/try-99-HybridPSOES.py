import numpy as np
import random

class HybridPSOES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_candidate = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = np.inf
        self.particles = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.velocity = np.random.uniform(-0.1, 0.1, (self.population_size, self.dim))
        self.w = 0.7
        self.c1 = 1.5
        self.c2 = 1.5

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = func(self.candidates[:, 0])
            self.best_candidate = self.candidates[np.argmin(self.candidates[:, 0]), :]
            self.best_fitness = fitness

            # Particle Swarm Optimization
            for i in range(self.population_size):
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)
                new_particle = self.particles[i] + self.velocity[i] + self.c1 * r1 * (self.best_candidate - self.particles[i]) + self.c2 * r2 * (self.candidates[np.argmin(self.candidates[:, 0]), :] - self.particles[i])
                new_fitness = func(new_particle)
                if new_fitness < self.best_fitness:
                    self.best_candidate = new_particle
                    self.best_fitness = new_fitness
                    self.candidates[np.argmin(self.candidates[:, 0]), :] = new_particle
                    self.particles[i] = new_particle
                    self.velocity[i] = self.w * self.velocity[i] + 0.1 * (self.best_candidate - self.particles[i])

            # Evolutionary Strategy
            for i in range(self.population_size):
                if random.random() < 0.25:
                    self.particles[i] += self.velocity[i]
                    self.particles[i] = np.clip(self.particles[i], -5.0, 5.0)

            # Selection
            self.candidates = self.candidates[np.argsort(self.candidates[:, 0])]
            self.population_size = self.population_size // 2

            # Mutation
            self.candidates[np.random.choice(self.population_size, size=self.population_size, replace=False), :] += np.random.uniform(-0.1, 0.1, size=(self.population_size, self.dim))

            # Check if the best candidate is improved
            if self.best_fitness < func(self.best_candidate):
                self.candidates[np.argmin(self.candidates[:, 0]), :] = self.best_candidate

        return self.best_candidate, self.best_fitness

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

hybrid_PSOES = HybridPSOES(budget=100, dim=2)
best_candidate, best_fitness = hybrid_PSOES(func)
print(f"Best candidate: {best_candidate}, Best fitness: {best_fitness}")