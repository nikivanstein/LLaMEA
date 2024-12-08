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
        self.prob_change = 0.05

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
                # Differential Evolution mutation
                v = np.copy(x)
                for k in range(self.dim):
                    v[k] = x[k] + self.c1 * (self.harmony_memory[np.random.randint(0, self.swarm_size), k] - self.harmony_memory[np.random.randint(0, self.swarm_size), k])
                fitness = func(v)
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = v
                self.harmony_memory[j] = x
                # Change individual lines with probability
                if random.random() < self.prob_change:
                    for k in range(self.dim):
                        r1 = random.random()
                        if r1 < 0.5:
                            self.best_solution[k] = self.harmony_memory[j, k]
                        else:
                            self.best_solution[k] = self.best_solution[k] + self.c2 * (self.harmony_memory[np.random.randint(0, self.swarm_size), k] - self.harmony_memory[np.random.randint(0, self.swarm_size), k])
            self.harmony_memory_size = int(self.harmony_memory_size * (1 - (i / self.max_iter)))
            self.harmony_memory = np.random.choice(self.harmony_memory, self.harmony_memory_size, replace=False)
            if i % 10 == 0:
                print(f"Iteration {i+1}, Best fitness: {self.best_fitness}")
        return self.best_solution