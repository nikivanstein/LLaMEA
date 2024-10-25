import numpy as np
import random

class HarmonyDE_SACPLR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.F = 0.5
        self.CR = 0.5
        self.sigma = 0.1
        self.learning_rate = 0.01
        self.crossover_probability = 0.5
        self.harmony_memory_size = 10
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.fitness = np.inf * np.ones(self.population_size)
        self.best_x = np.inf * np.ones(self.dim)
        self.best_fitness = np.inf

    def __call__(self, func):
        for i in range(self.budget):
            y = func(self.x)
            self.fitness = y
            idx = np.argmin(y)
            self.best_x = self.x[idx]
            self.best_fitness = y[idx]
            for j in range(self.population_size):
                if j!= idx:
                    r1, r2, r3 = random.sample(range(self.population_size), 3)
                    while r1 == idx or r2 == idx or r3 == idx:
                        r1, r2, r3 = random.sample(range(self.population_size), 3)
                    x_new = self.x[r1] + self.F * (self.x[r2] - self.x[r3])
                    x_new = x_new + self.sigma * np.random.normal(0, 1, self.dim)
                    x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
                    y_new = func(x_new)
                    if y_new < self.fitness[j]:
                        self.x[j] = x_new
                        self.fitness[j] = y_new
            # Update harmony memory
            self.harmony_memory = np.vstack((self.harmony_memory, self.x[idx]))
            self.harmony_memory = self.harmony_memory[np.random.choice(self.harmony_memory.shape[0], self.harmony_memory_size, replace=False), :]
            # Update CR and sigma
            self.CR = self.CR + self.learning_rate * (self.crossover_probability - self.CR)
            self.crossover_probability = max(0.1, min(1.0, self.CR))
            self.sigma = self.sigma + self.learning_rate * (self.sigma - self.fitness[idx])
            # Update best solution
            if self.fitness[idx] < self.best_fitness:
                self.best_fitness = self.fitness[idx]
                self.best_x = self.x[idx]
        # Select new individual lines based on probability
        select_prob = 0.06521739130434782
        new_individual = self.best_x.copy()
        for i in range(self.dim):
            if random.random() < select_prob:
                new_individual[i] = np.random.uniform(self.lower_bound, self.upper_bound)
        return new_individual, self.best_fitness