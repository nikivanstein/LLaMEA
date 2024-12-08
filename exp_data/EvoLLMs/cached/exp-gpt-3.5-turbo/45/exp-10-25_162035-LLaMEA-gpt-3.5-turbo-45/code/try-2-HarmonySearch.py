import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.hmcr = 0.7
        self.par = 0.3
        self.bandwidth = 0.01
        self.harmonies = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        self.fitness = np.full(self.budget, np.inf)

    def __call__(self, func):
        for t in range(self.budget):
            new_solution = np.zeros((1, self.dim))
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_solution[0, j] = self.harmonies[np.random.randint(self.budget), j]
                else:
                    new_solution[0, j] = self.lower_bound + np.random.rand() * (self.upper_bound - self.lower_bound)
                if np.random.rand() < self.par:
                    new_solution[0, j] += np.random.uniform(-self.bandwidth, self.bandwidth)
            new_fitness = func(new_solution)
            if new_fitness < np.max(self.fitness):
                self.fitness[np.argmax(self.fitness)] = new_fitness
                self.harmonies[np.argmax(self.fitness)] = new_solution
            if np.random.rand() < 0.35:
                self.hmcr = np.clip(self.hmcr + np.random.uniform(-0.05, 0.05), 0.1, 0.9)
                self.par = np.clip(self.par + np.random.uniform(-0.05, 0.05), 0.1, 0.9)
                self.bandwidth = np.clip(self.bandwidth + np.random.uniform(-0.002, 0.002), 0.001, 0.02)
        return self.harmonies[np.argmin(self.fitness)]