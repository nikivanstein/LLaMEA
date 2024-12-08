import numpy as np
import random

class DEHS:
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
        self.probability = 0.05

    def __call__(self, func):
        for i in range(self.max_iter):
            for j in range(self.swarm_size):
                x = np.copy(self.harmony_memory[j])
                for k in range(self.dim):
                    r1 = random.random()
                    if r1 < self.probability:
                        x[k] = self.harmony_memory[np.random.randint(0, self.harmony_memory_size), k] + np.random.uniform(-1, 1)
                    elif r1 < 0.3:
                        x[k] = self.best_solution[k] + np.random.uniform(-1, 1)
                    else:
                        r2 = random.randint(0, self.swarm_size - 1)
                        r3 = random.randint(0, self.swarm_memory_size - 1)
                        x[k] = self.harmony_memory[r2, k] + self.w * (self.harmony_memory[r3, k] - self.harmony_memory[np.random.randint(0, self.harmony_memory_size), k])
                fitness = func(x)
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = x
                self.harmony_memory[j] = x
            self.harmony_memory_size = int(self.harmony_memory_size * (1 - (i / self.max_iter)))
            self.harmony_memory = np.random.choice(self.harmony_memory, self.harmony_memory_size, replace=False)
            if i % 10 == 0:
                print(f"Iteration {i+1}, Best fitness: {self.best_fitness}")
            if np.sum(np.isinf(self.harmony_memory)) > 0 or np.sum(np.isnan(self.harmony_memory)) > 0:
                self.harmony_memory = np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))
        return self.best_solution