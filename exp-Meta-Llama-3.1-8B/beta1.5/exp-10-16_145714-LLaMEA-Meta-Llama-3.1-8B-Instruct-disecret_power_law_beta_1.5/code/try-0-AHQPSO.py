import numpy as np
import random

class AHQPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory = np.random.uniform(-5, 5, (budget, dim))
        self.velocity = np.random.uniform(-1, 1, (budget, dim))
        self.best_solution = np.zeros(dim)
        self.best_fitness = np.inf

    def __call__(self, func):
        for i in range(self.budget):
            fitness = func(self.harmony_memory[i])
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = self.harmony_memory[i]
            # Harmony Search
            if random.random() < 0.5:
                self.harmony_memory[i] = self.harmony_memory[i] + np.random.uniform(-1, 1, self.dim)
                self.harmony_memory[i] = np.clip(self.harmony_memory[i], -5, 5)
            # Quantum-Behaved Particle Swarm Optimization
            else:
                self.velocity[i] = 0.7298 * self.velocity[i] + 1.49618 * (self.best_solution - self.harmony_memory[i]) + 1.49618 * (np.random.uniform(-1, 1, self.dim))
                self.harmony_memory[i] = self.harmony_memory[i] + self.velocity[i]
                self.harmony_memory[i] = np.clip(self.harmony_memory[i], -5, 5)
        return self.best_solution, self.best_fitness

# Example usage:
def sphere(x):
    return np.sum(x**2)

budget = 1000
dim = 10
ahqspo = AHQPSO(budget, dim)
best_solution, best_fitness = ahqspo(sphere)
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)