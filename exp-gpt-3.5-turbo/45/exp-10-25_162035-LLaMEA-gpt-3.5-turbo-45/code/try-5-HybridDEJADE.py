import numpy as np

class HybridDEJADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.cr = 0.5
        self.f = 0.5
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        self.fitness = np.full(self.budget, np.inf)

    def __call__(self, func):
        for t in range(self.budget):
            for i in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = self.population[a] + self.f * (self.population[b] - self.population[c])
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, self.population[i])
                new_fitness = func(trial[np.newaxis, :])
                if new_fitness < self.fitness[i]:
                    self.fitness[i] = new_fitness
                    self.population[i] = trial
        return self.population[np.argmin(self.fitness)]