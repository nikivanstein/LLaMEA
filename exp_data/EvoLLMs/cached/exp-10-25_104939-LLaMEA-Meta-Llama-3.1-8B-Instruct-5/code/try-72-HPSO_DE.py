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
        self.DE_population_size = 50
        self.DE_dim = dim
        self.DE_F = 0.5
        self.DE_CR = 0.9

    def __call__(self, func):
        for i in range(self.max_iter):
            for j in range(self.swarm_size):
                x = np.random.uniform(-5.0, 5.0, self.dim)
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

            # Differential Evolution
            DE_population = np.random.uniform(-5.0, 5.0, (self.DE_population_size, self.DE_dim))
            for k in range(self.DE_population_size):
                x1 = DE_population[k]
                x2 = DE_population[np.random.randint(0, self.DE_population_size)]
                x3 = DE_population[np.random.randint(0, self.DE_population_size)]
                x4 = DE_population[np.random.randint(0, self.DE_population_size)]
                v = x1 + self.DE_F * (x2 - x3) + self.DE_CR * (x4 - x1)
                v = np.clip(v, -5.0, 5.0)
                fitness = func(v)
                if fitness < func(x1):
                    DE_population[k] = v
            if np.any(DE_population < -5.0) or np.any(DE_population > 5.0):
                print("Warning: DE population out of bounds.")
            if func(DE_population[np.argmin([func(x) for x in DE_population])]) < self.best_fitness:
                self.best_fitness = func(DE_population[np.argmin([func(x) for x in DE_population])])
                self.best_solution = DE_population[np.argmin([func(x) for x in DE_population])]

            if i % 10 == 0:
                print(f"Iteration {i+1}, Best fitness: {self.best_fitness}")
        return self.best_solution