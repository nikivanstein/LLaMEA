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
        self.DE_strategy = np.random.choice(['rand/1/bin', 'best/1/bin', 'rand/2/bin'], p=[0.05, 0.45, 0.5])

    def __call__(self, func):
        for i in range(self.max_iter):
            for j in range(self.swarm_size):
                x = np.random.uniform(-5.0, 5.0, self.dim)
                if np.random.rand() < 0.05:
                    strategy = random.choice(['rand/1/bin', 'best/1/bin', 'rand/2/bin'])
                    x = self.differential_evolution(x, func, strategy)
                else:
                    harmony = np.random.uniform(-5.0, 5.0, self.dim)
                    for k in range(self.dim):
                        r1 = random.random()
                        r2 = random.random()
                        if r1 < 0.5:
                            x[k] = self.harmony_memory[j, k] + self.w * (self.harmony_memory[j, k] - self.harmony_memory[np.random.randint(0, self.harmony_memory_size), k])
                        else:
                            x[k] = self.best_solution[k] + self.w * (self.best_solution[k] - self.harmony_memory[np.random.randint(0, self.harmony_memory_size), k])
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

    def differential_evolution(self, x, func, strategy):
        F = 0.5
        CR = 0.9
        NP = 50
        x1 = np.random.uniform(-5.0, 5.0, self.dim)
        x2 = np.random.uniform(-5.0, 5.0, self.dim)
        x3 = np.random.uniform(-5.0, 5.0, self.dim)
        if strategy == 'rand/1/bin':
            u = np.random.randint(0, NP)
            v = np.random.randint(0, NP)
            while u == v:
                v = np.random.randint(0, NP)
            x_new = x + F * (x2 - x1)
        elif strategy == 'best/1/bin':
            x_new = x + F * (self.best_solution - x)
        elif strategy == 'rand/2/bin':
            u = np.random.randint(0, NP)
            v = np.random.randint(0, NP)
            while u == v:
                v = np.random.randint(0, NP)
            x_new = x + F * (x2 - x1) + F * (x3 - x2)
        for k in range(self.dim):
            r = np.random.rand()
            if r < CR or k == np.random.randint(0, self.dim):
                x_new[k] = np.random.uniform(-5.0, 5.0)
        fitness = func(x_new)
        return x_new