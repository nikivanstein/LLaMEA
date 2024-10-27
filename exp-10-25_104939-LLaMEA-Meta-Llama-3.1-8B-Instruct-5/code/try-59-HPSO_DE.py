import numpy as np
import random

class HPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 10
        self.max_iter = 100
        self.swarm_size = 50
        self.w = 0.7298
        self.c1 = 1.49618
        self.c2 = 1.49618
        self.harmony_memory = np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))
        self.best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = float('inf')
        self.population = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        self.fitness = np.zeros(self.swarm_size)

    def __call__(self, func):
        for i in range(self.max_iter):
            for j in range(self.swarm_size):
                x = np.copy(self.population[j])
                for k in range(self.dim):
                    if random.random() < 0.05:
                        x[k] = self.population[np.random.randint(0, self.swarm_size), k] + random.uniform(-1, 1)
                fitness = func(x)
                if fitness < self.fitness[j]:
                    self.fitness[j] = fitness
                    self.population[j] = x
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = x
            self.harmony_memory_size = int(self.harmony_memory_size * (1 - (i / self.max_iter)))
            self.harmony_memory = np.random.choice(self.harmony_memory, self.harmony_memory_size, replace=False)
            for j in range(self.swarm_size):
                x = np.copy(self.population[j])
                for k in range(self.dim):
                    r1 = random.randint(0, self.swarm_size - 1)
                    r2 = random.randint(0, self.swarm_size - 1)
                    r3 = random.randint(0, self.swarm_size - 1)
                    x[k] = x[k] + self.c1 * (self.population[r1, k] - x[k]) + self.c2 * (self.population[r2, k] - self.population[r3, k])
                self.population[j] = x
            if i % 10 == 0:
                print(f"Iteration {i+1}, Best fitness: {self.best_fitness}")
        return self.best_solution