import numpy as np
import random

class HPCSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.max_iter = 100
        self.swarm_size = 50
        self.w = 0.7298
        self.c1 = 1.49618
        self.c2 = 1.49618
        self.least_fitness = float('inf')
        self.best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = float('inf')
        self.cuckoo_eggs = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        self.probability = 0.05

    def __call__(self, func):
        for i in range(self.max_iter):
            for j in range(self.swarm_size):
                x = np.random.uniform(-5.0, 5.0, self.dim)
                for k in range(self.dim):
                    r1 = random.random()
                    if r1 < self.probability:
                        x[k] = self.cuckoo_eggs[j, k] + np.random.uniform(-1, 1)
                fitness = func(x)
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = x
                if fitness < self.least_fitness:
                    self.least_fitness = fitness
                    self.cuckoo_eggs[j] = x
                else:
                    if random.random() < 0.25:
                        self.cuckoo_eggs[j] = x
            if i % 10 == 0:
                print(f"Iteration {i+1}, Best fitness: {self.best_fitness}")
            if self.budget > 0:
                self.budget -= 1
            else:
                break
        return self.best_solution