import numpy as np

class HarmonySearch_SimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.harmony_memory_size = 10
        self.para = 0.5
        self.max_iter = 100
        self.t = 100
        self.alpha = 0.99
        self.beta = 0.01
        self.x = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.inf * np.ones(self.population_size)
        self.best_x = np.inf * np.ones(self.dim)
        self.best_fitness = np.inf

    def __call__(self, func):
        for i in range(self.budget):
            y = func(self.x)
            self.fitness = y
            idx = np.argmin(y)
            self.best_x = self.x[idx]
            self.best_fitness = y[idx]
            for j in range(self.population_size):
                if j!= idx:
                    r1, r2, r3 = random.sample(range(self.population_size), 3)
                    while r1 == idx or r2 == idx or r3 == idx:
                        r1, r2, r3 = random.sample(range(self.population_size), 3)
                    x_new = self.x[r1] + (self.x[r2] - self.x[r3]) * np.random.uniform(-1, 1)
                    x_new = x_new + np.random.normal(0, 1, self.dim)
                    x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
                    y_new = func(x_new)
                    if y_new < self.fitness[j]:
                        self.x[j] = x_new
                        self.fitness[j] = y_new
            # Harmony Search
            for j in range(self.harmony_memory_size):
                x_new = np.mean(self.x, axis=0) + np.random.uniform(-1, 1) * np.std(self.x, axis=0)
                x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
                y_new = func(x_new)
                if y_new < self.fitness[j]:
                    self.x[j] = x_new
                    self.fitness[j] = y_new
            # Simulated Annealing
            for j in range(self.population_size):
                if np.random.uniform(0, 1) < self.para:
                    r1, r2 = random.sample(range(self.population_size), 2)
                    while r1 == j or r2 == j:
                        r1, r2 = random.sample(range(self.population_size), 2)
                    x_new = self.x[r1] + (self.x[r2] - self.x[j]) * np.random.uniform(-1, 1)
                    x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
                    y_new = func(x_new)
                    if y_new < self.fitness[j]:
                        self.x[j] = x_new
                        self.fitness[j] = y_new
                if self.fitness[j] < self.best_fitness:
                    self.best_fitness = self.fitness[j]
                    self.best_x = self.x[j]
            self.para = self.alpha * self.para + self.beta * (1 - self.para)
            self.t = self.t * self.alpha
        return self.best_x, self.best_fitness