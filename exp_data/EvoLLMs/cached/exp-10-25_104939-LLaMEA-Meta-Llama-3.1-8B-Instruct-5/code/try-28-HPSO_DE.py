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
        self.DE_strategy = np.random.choice(['rand/1/bin', 'best/1/bin', 'rand/2/bin', 'best/2/bin'], p=[0.05, 0.05, 0.45, 0.45])

    def __call__(self, func):
        for i in range(self.max_iter):
            for j in range(self.swarm_size):
                x = np.random.uniform(-5.0, 5.0, self.dim)
                if np.random.random() < 0.05 and self.DE_strategy == 'rand/1/bin':
                    x = self.rand_1_bin(x, self.harmony_memory[j], self.best_solution)
                elif np.random.random() < 0.05 and self.DE_strategy == 'best/1/bin':
                    x = self.best_1_bin(x, self.harmony_memory[j], self.best_solution)
                elif np.random.random() < 0.05 and self.DE_strategy == 'rand/2/bin':
                    x = self.rand_2_bin(x, self.harmony_memory[j], self.best_solution)
                elif np.random.random() < 0.05 and self.DE_strategy == 'best/2/bin':
                    x = self.best_2_bin(x, self.harmony_memory[j], self.best_solution)
                else:
                    r1 = random.random()
                    r2 = random.random()
                    if r1 < 0.5:
                        x = self.harmony_memory[j] + self.w * (self.harmony_memory[j] - self.harmony_memory[np.random.randint(0, self.harmony_memory_size), :])
                    else:
                        x = self.best_solution + self.w * (self.best_solution - self.harmony_memory[np.random.randint(0, self.harmony_memory_size), :])
                fitness = func(x)
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = x
                self.harmony_memory[j] = x
            self.harmony_memory_size = int(self.harmony_memory_size * (1 - (i / self.max_iter)))
            self.harmony_memory = np.random.choice(self.harmony_memory, self.harmony_memory_size, replace=False)
            self.DE_strategy = np.random.choice(['rand/1/bin', 'best/1/bin', 'rand/2/bin', 'best/2/bin'], p=[0.05, 0.05, 0.45, 0.45])
            if i % 10 == 0:
                print(f"Iteration {i+1}, Best fitness: {self.best_fitness}")
        return self.best_solution

    def rand_1_bin(self, x, x1, x2):
        F = 0.5
        v = F * (x2 - x1) + x1
        u = np.random.uniform(-1, 1, self.dim)
        return x + u * (v - x)

    def best_1_bin(self, x, x1, x2):
        F = 0.5
        v = F * (x2 - x) + x
        u = np.random.uniform(-1, 1, self.dim)
        return x + u * (v - x)

    def rand_2_bin(self, x, x1, x2):
        F = 0.5
        v1 = F * (x2 - x1) + x1
        v2 = F * (x2 - x1) + x1
        u = np.random.uniform(-1, 1, self.dim)
        return x + u[0] * (v1 - x) + u[1] * (v2 - x)

    def best_2_bin(self, x, x1, x2):
        F = 0.5
        v1 = F * (x2 - x) + x
        v2 = F * (x2 - x) + x
        u = np.random.uniform(-1, 1, self.dim)
        return x + u[0] * (v1 - x) + u[1] * (v2 - x)