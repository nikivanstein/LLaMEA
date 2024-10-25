import numpy as np

class BeesAlgorithm:
    def __init__(self, budget, dim, colony_size=10, elite_sites=3, patch_size=3, neighborhood_size=5):
        self.budget = budget
        self.dim = dim
        self.colony_size = colony_size
        self.elite_sites = elite_sites
        self.patch_size = patch_size
        self.neighborhood_size = neighborhood_size
        self.colony = np.random.uniform(-5.0, 5.0, (colony_size, dim))
        self.best_solution = np.copy(self.colony[0])

    def __call__(self, func):
        for t in range(self.budget):
            for i in range(self.colony_size):
                site = np.random.randint(self.elite_sites)
                patch = self.colony[np.random.choice(self.colony_size, self.patch_size, replace=False)]
                new_solution = self.colony[i] + np.mean(patch, axis=0) - self.colony[i]
                if func(new_solution) < func(self.colony[i]):
                    self.colony[i] = new_solution
            neighborhoods = [np.random.choice(self.colony_size, self.neighborhood_size, replace=False) for _ in range(self.colony_size)]
            for i in range(self.colony_size):
                best_neighbor = min(neighborhoods[i], key=lambda x: func(self.colony[x]))
                if func(self.colony[best_neighbor]) < func(self.best_solution):
                    self.best_solution = np.copy(self.colony[best_neighbor])
        return self.best_solution