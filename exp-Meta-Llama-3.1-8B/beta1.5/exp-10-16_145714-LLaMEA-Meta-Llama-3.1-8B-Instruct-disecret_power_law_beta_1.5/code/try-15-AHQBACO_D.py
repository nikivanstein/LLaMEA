import numpy as np
import random

class AHQBACO_D:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory = np.random.uniform(-5, 5, (budget, dim))
        self.velocity = np.random.uniform(-1, 1, (budget, dim))
        self.best_solution = np.zeros(dim)
        self.best_fitness = np.inf
        self.employed_bees = np.random.uniform(-5, 5, (budget, dim))
        self.onlooker_bees = np.random.uniform(-5, 5, (budget, dim))
        self.dance_memory = np.zeros((budget, dim))
        self.diversity = np.mean(np.linalg.norm(self.harmony_memory, axis=1))

    def __call__(self, func):
        for i in range(self.budget):
            fitness = func(self.harmony_memory[i])
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = self.harmony_memory[i]
            # Dynamic exploration rate
            exploration_rate = 0.5 * (1 - self.diversity) + 0.5
            # Harmony Search
            if random.random() < exploration_rate:
                self.harmony_memory[i] = self.harmony_memory[i] + np.random.uniform(-1, 1, self.dim)
                self.harmony_memory[i] = np.clip(self.harmony_memory[i], -5, 5)
            # Quantum-Behaved Particle Swarm Optimization
            else:
                self.velocity[i] = 0.7298 * self.velocity[i] + 1.49618 * (self.best_solution - self.harmony_memory[i]) + 1.49618 * (np.random.uniform(-1, 1, self.dim))
                self.harmony_memory[i] = self.harmony_memory[i] + self.velocity[i]
                self.harmony_memory[i] = np.clip(self.harmony_memory[i], -5, 5)
            # Artificial Bee Colony Optimization
            if i < self.budget // 2:
                # Employed bees
                self.employed_bees[i] = self.harmony_memory[i] + np.random.uniform(-1, 1, self.dim)
                self.employed_bees[i] = np.clip(self.employed_bees[i], -5, 5)
                # Onlooker bees
                self.onlooker_bees[i] = self.dance_memory[i] + np.random.uniform(-1, 1, self.dim)
                self.onlooker_bees[i] = np.clip(self.onlooker_bees[i], -5, 5)
                # Dance memory
                self.dance_memory[i] = self.onlooker_bees[i] if func(self.onlooker_bees[i]) < func(self.dance_memory[i]) else self.dance_memory[i]
                # Update diversity
                self.diversity = np.mean(np.linalg.norm(self.harmony_memory, axis=1))
            else:
                # Employed bees
                self.employed_bees[i] = self.onlooker_bees[i] + np.random.uniform(-1, 1, self.dim)
                self.employed_bees[i] = np.clip(self.employed_bees[i], -5, 5)
                # Onlooker bees
                self.onlooker_bees[i] = self.dance_memory[i] + np.random.uniform(-1, 1, self.dim)
                self.onlooker_bees[i] = np.clip(self.onlooker_bees[i], -5, 5)
                # Dance memory
                self.dance_memory[i] = self.onlooker_bees[i] if func(self.onlooker_bees[i]) < func(self.dance_memory[i]) else self.dance_memory[i]
                # Update diversity
                self.diversity = np.mean(np.linalg.norm(self.harmony_memory, axis=1))
        return self.best_solution, self.best_fitness

# Example usage:
def sphere(x):
    return np.sum(x**2)

budget = 1000
dim = 10
ahqbaco_d = AHQBACO_D(budget, dim)
best_solution, best_fitness = ahqbaco_d(sphere)
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)