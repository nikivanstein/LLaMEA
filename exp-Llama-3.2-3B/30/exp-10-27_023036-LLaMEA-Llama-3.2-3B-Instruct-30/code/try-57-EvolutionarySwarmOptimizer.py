import numpy as np
import random

class EvolutionarySwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.particles = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.pbest = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.gbest = np.random.uniform(-5.0, 5.0, self.dim)
        self.candidates = []
        self.fitness_values = []

    def __call__(self, func):
        for _ in range(self.budget):
            for particle in self.particles:
                fitness = func(particle)
                self.fitness_values.append(fitness)
                if fitness < func(self.pbest[np.argmin(self.fitness_values)]):
                    self.pbest[np.argmin(self.fitness_values)] = particle
                if fitness < func(self.gbest):
                    self.gbest = particle
            self.candidates.append(self.pbest[np.argmin(self.fitness_values)])
            self.fitness_values = []
            # Evolutionary strategy: perturb the best particle
            if random.random() < 0.3:
                self.pbest[np.argmin(self.fitness_values)] += np.random.uniform(-0.1, 0.1, self.dim)
                self.fitness_values.append(func(self.pbest[np.argmin(self.fitness_values)]))
            # Particle swarm optimization: update the best particle
            if random.random() < 0.3:
                self.pbest[np.argmin(self.fitness_values)] = self.gbest + np.random.uniform(-0.1, 0.1, self.dim)
                self.fitness_values.append(func(self.pbest[np.argmin(self.fitness_values)]))
            # Random mutation
            if random.random() < 0.3:
                self.pbest[np.argmin(self.fitness_values)] += np.random.uniform(-0.1, 0.1, self.dim)
                self.fitness_values.append(func(self.pbest[np.argmin(self.fitness_values)]))
        return self.candidates[np.argmin(self.fitness_values)]

# Example usage
def func(x):
    return np.sum(x**2)

optimizer = EvolutionarySwarmOptimizer(100, 2)
best_solution = optimizer(func)
print(best_solution)