import numpy as np
import random

class HSA_QEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.HS_rate = 0.01
        self.QEA_rate = 0.1
        self.x = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.inf * np.ones(self.population_size)
        self.best_x = np.inf * np.ones(self.dim)
        self.best_fitness = np.inf
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.quantum_bit_string = np.random.choice([0, 1], size=(self.population_size, self.dim))

    def __call__(self, func):
        for i in range(self.budget):
            y = func(self.x)
            self.fitness = y
            idx = np.argmin(y)
            self.best_x = self.x[idx]
            self.best_fitness = y[idx]
            for j in range(self.population_size):
                if random.random() < self.HS_rate:
                    r1 = random.randint(0, self.population_size - 1)
                    x_new = self.harmony_memory[r1] + np.random.uniform(-1, 1, self.dim) * (self.x[r1] - self.harmony_memory[r1])
                    x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
                    y_new = func(x_new)
                    if y_new < self.fitness[j]:
                        self.x[j] = x_new
                        self.fitness[j] = y_new
                        self.harmony_memory[j] = x_new
                else:
                    for k in range(self.dim):
                        if random.random() < self.QEA_rate:
                            self.quantum_bit_string[j, k] = 1 - self.quantum_bit_string[j, k]
                    x_new = np.zeros(self.dim)
                    for k in range(self.dim):
                        if self.quantum_bit_string[j, k] == 1:
                            x_new[k] = self.x[j, k] + np.random.uniform(-1, 1, 1)[0]
                        else:
                            x_new[k] = self.x[j, k] - np.random.uniform(-1, 1, 1)[0]
                    x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
                    y_new = func(x_new)
                    if y_new < self.fitness[j]:
                        self.x[j] = x_new
                        self.fitness[j] = y_new
            self.HS_rate = self.HS_rate + self.HS_rate * (self.HS_rate - 0.01)
            self.HS_rate = max(0.001, min(0.1, self.HS_rate))
            self.QEA_rate = self.QEA_rate + self.QEA_rate * (self.QEA_rate - 0.1)
            self.QEA_rate = max(0.001, min(0.5, self.QEA_rate))
        return self.best_x, self.best_fitness