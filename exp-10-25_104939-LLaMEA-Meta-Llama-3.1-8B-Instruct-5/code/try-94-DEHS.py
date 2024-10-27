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

    def __call__(self, func):
        for i in range(self.max_iter):
            for j in range(self.swarm_size):
                x = np.copy(self.harmony_memory[j])
                for k in range(self.dim):
                    r1, r2, r3 = random.sample(range(self.swarm_size), 3)
                    if random.random() < 0.05:
                        x[k] = self.harmony_memory[np.random.randint(0, self.harmony_memory_size), k] + np.random.uniform(-1, 1)
                    else:
                        x[k] = self.harmony_memory[r1, k] + self.w * (self.harmony_memory[r2, k] - self.harmony_memory[r3, k])
                x = np.clip(x, -5.0, 5.0)
                fitness = func(x)
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = x
                self.harmony_memory[j] = x
            self.harmony_memory_size = int(self.harmony_memory_size * (1 - (i / self.max_iter)))
            self.harmony_memory = np.random.choice(self.harmony_memory, self.harmony_memory_size, replace=False)
            if i % 10 == 0:
                print(f"Iteration {i+1}, Best fitness: {self.best_fitness}")
        return self.best_solution