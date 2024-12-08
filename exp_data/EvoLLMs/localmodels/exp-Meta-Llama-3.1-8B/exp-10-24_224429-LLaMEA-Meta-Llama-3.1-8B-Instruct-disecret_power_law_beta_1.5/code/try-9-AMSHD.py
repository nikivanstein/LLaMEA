import numpy as np
import random

class AMSHD:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.harmony_memory_size = int(0.2 * self.population_size)
        self.swarm_size = int(0.8 * self.population_size)
        self.dynamic_neighborhood_size = int(0.1 * self.population_size)
        self.parsimony_coefficient = 0.01
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.swarms = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        self.fitness = np.inf * np.ones(self.population_size)
        self.best_x = np.inf * np.ones(self.dim)
        self.best_fitness = np.inf

    def __call__(self, func):
        for i in range(self.budget):
            y = func(self.harmony_memory)
            self.fitness[:self.harmony_memory_size] = y
            idx = np.argmin(y)
            self.best_x = self.harmony_memory[idx]
            self.best_fitness = y[idx]
            for j in range(self.population_size):
                if j < self.harmony_memory_size:
                    continue
                swarm_idx = random.randint(0, self.swarm_size - 1)
                r1, r2 = random.sample(range(self.population_size), 2)
                while r1 == j or r2 == j:
                    r1, r2 = random.sample(range(self.population_size), 2)
                x_new = self.swarms[swarm_idx] + np.random.uniform(-1, 1, self.dim) * (self.swarms[swarm_idx] - self.swarms[random.randint(0, self.swarm_size - 1)])
                x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
                y_new = func(x_new)
                if y_new < self.fitness[j]:
                    self.swarms[swarm_idx] = x_new
                    self.fitness[j] = y_new
            # Dynamic Neighborhoods
            for j in range(self.population_size):
                if j < self.harmony_memory_size:
                    continue
                r1, r2, r3 = random.sample(range(self.population_size), 3)
                while r1 == j or r2 == j or r3 == j:
                    r1, r2, r3 = random.sample(range(self.population_size), 3)
                x_new = self.swarms[random.randint(0, self.swarm_size - 1)] + np.random.uniform(-1, 1, self.dim) * (self.swarms[random.randint(0, self.swarm_size - 1)] - self.swarms[r1])
                x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
                y_new = func(x_new)
                if y_new < self.fitness[j]:
                    self.swarms[random.randint(0, self.swarm_size - 1)] = x_new
                    self.fitness[j] = y_new
            # Update Harmony Memory
            for j in range(self.harmony_memory_size):
                r1 = random.randint(0, self.population_size - 1)
                while r1 == j:
                    r1 = random.randint(0, self.population_size - 1)
                if self.fitness[r1] < self.fitness[j]:
                    self.harmony_memory[j] = self.swarms[random.randint(0, self.swarm_size - 1)]
                    self.fitness[j] = self.fitness[r1]
            # Update Best Solution
            if self.fitness[j] < self.best_fitness:
                self.best_fitness = self.fitness[j]
                self.best_x = self.swarms[random.randint(0, self.swarm_size - 1)]
        return self.best_x, self.best_fitness