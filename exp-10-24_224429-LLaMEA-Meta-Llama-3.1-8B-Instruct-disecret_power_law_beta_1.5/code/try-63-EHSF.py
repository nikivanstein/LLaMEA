import numpy as np

class EHSF:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.harmony_memory_size = 10
        self.max_frequency = 10.0
        self.min_frequency = 0.1
        self.paranoid = 0.5
        self.x = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.inf * np.ones(self.population_size)
        self.best_x = np.inf * np.ones(self.dim)
        self.best_fitness = np.inf
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))

    def __call__(self, func):
        for i in range(self.budget):
            y = func(self.x)
            self.fitness = y
            idx = np.argmin(y)
            self.best_x = self.x[idx]
            self.best_fitness = y[idx]
            for j in range(self.population_size):
                if j!= idx:
                    r1, r2 = random.sample(range(self.population_size), 2)
                    while r1 == idx or r2 == idx:
                        r1, r2 = random.sample(range(self.population_size), 2)
                    x_new = self.x[r1] + np.random.uniform(-1, 1, self.dim)
                    x_new = x_new + self.paranoid * np.random.normal(0, 1, self.dim)
                    x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
                    y_new = func(x_new)
                    if y_new < self.fitness[j]:
                        self.x[j] = x_new
                        self.fitness[j] = y_new
            # Update harmony memory
            self.harmony_memory = np.vstack((self.harmony_memory, self.x[idx]))
            self.harmony_memory = np.delete(self.harmony_memory, 0, 0)
            # Update frequency
            self.max_frequency = self.max_frequency + 0.01 * (self.max_frequency - self.fitness[idx])
            self.min_frequency = self.min_frequency + 0.01 * (self.min_frequency - self.fitness[idx])
            self.paranoid = self.paranoid + 0.01 * (self.paranoid - self.fitness[idx])
            if self.fitness[idx] < self.best_fitness:
                self.best_fitness = self.fitness[idx]
                self.best_x = self.x[idx]
        return self.best_x, self.best_fitness