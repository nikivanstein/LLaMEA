import numpy as np
import random

class HyperEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.fitness = np.zeros(budget)
        self.fitness_argsort = np.argsort(self.fitness)
        self.best_idx = 0
        self.perturb_prob = 0.3

    def __call__(self, func):
        for i in range(self.budget):
            idx = self.fitness_argsort[i]
            self.fitness[idx] = func(self.population[idx])
            if i < self.budget - 1:
                if random.random() < self.perturb_prob:
                    self.population[idx] += np.random.uniform(-0.1, 0.1, self.dim)
                    self.fitness[idx] = func(self.population[idx])
                else:
                    idx2 = np.random.choice(self.fitness_argsort[:i+1])
                    self.population[idx], self.population[idx2] = self.population[idx2], self.population[idx]
                    self.fitness[idx], self.fitness[idx2] = self.fitness[idx2], self.fitness[idx]

        self.population = self.population[self.fitness_argsort]
        self.fitness = self.fitness[self.fitness_argsort]
        self.best_idx = np.argmax(self.fitness)
        self.population = self.population[self.best_idx]
        self.fitness = self.fitness[self.best_idx]

# Example usage:
def func(x):
    return np.sum(x**2)

evolution = HyperEvolution(budget=100, dim=10)
best_x = evolution(func)
print(best_x)