import numpy as np
import random

class AHQBACO_DN2:
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
        self.neighborhood_size = 0.2 * budget  # dynamic neighborhood size
        self.exploration_rate = 0.5  # dynamic exploration rate
        self.convergence_threshold = 0.1  # convergence threshold

    def __call__(self, func):
        convergence = np.zeros(self.budget)
        for i in range(self.budget):
            fitness = func(self.harmony_memory[i])
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = self.harmony_memory[i]
            # Harmony Search
            if random.random() < self.exploration_rate:
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
                if random.random() < 0.5:
                    self.onlooker_bees[i] = self.employed_bees[i] + np.random.uniform(-self.neighborhood_size, self.neighborhood_size, self.dim)
                    self.onlooker_bees[i] = np.clip(self.onlooker_bees[i], -5, 5)
                else:
                    self.onlooker_bees[i] = self.dance_memory[i] + np.random.uniform(-1, 1, self.dim)
                    self.onlooker_bees[i] = np.clip(self.onlooker_bees[i], -5, 5)
                # Dance memory
                self.dance_memory[i] = self.onlooker_bees[i] if func(self.onlooker_bees[i]) < func(self.dance_memory[i]) else self.dance_memory[i]
                convergence[i] = np.linalg.norm(self.harmony_memory[i] - self.best_solution)
            else:
                # Employed bees
                self.employed_bees[i] = self.onlooker_bees[i] + np.random.uniform(-1, 1, self.dim)
                self.employed_bees[i] = np.clip(self.employed_bees[i], -5, 5)
                # Onlooker bees
                if random.random() < 0.5:
                    self.onlooker_bees[i] = self.employed_bees[i] + np.random.uniform(-self.neighborhood_size, self.neighborhood_size, self.dim)
                    self.onlooker_bees[i] = np.clip(self.onlooker_bees[i], -5, 5)
                else:
                    self.onlooker_bees[i] = self.dance_memory[i] + np.random.uniform(-1, 1, self.dim)
                    self.onlooker_bees[i] = np.clip(self.onlooker_bees[i], -5, 5)
                # Dance memory
                self.dance_memory[i] = self.onlooker_bees[i] if func(self.onlooker_bees[i]) < func(self.dance_memory[i]) else self.dance_memory[i]
                convergence[i] = np.linalg.norm(self.harmony_memory[i] - self.best_solution)
            if np.mean(convergence) < self.convergence_threshold:
                self.exploration_rate = max(0.1, self.exploration_rate * 0.9)
            else:
                self.exploration_rate = min(0.9, self.exploration_rate * 1.1)
            self.neighborhood_size = max(0.1 * budget, self.neighborhood_size * 0.9)  # adjust neighborhood size based on fitness of employed bees
        return self.best_solution, self.best_fitness

# Example usage:
def sphere(x):
    return np.sum(x**2)

budget = 1000
dim = 10
ahqbaco_dn2 = AHQBACO_DN2(budget, dim)
best_solution, best_fitness = ahqbaco_dn2(sphere)
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)