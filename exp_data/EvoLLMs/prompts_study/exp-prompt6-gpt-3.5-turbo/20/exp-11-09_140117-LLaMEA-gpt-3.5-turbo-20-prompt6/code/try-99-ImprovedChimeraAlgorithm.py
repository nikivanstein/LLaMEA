import numpy as np

class ImprovedChimeraAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.crossover_prob = 0.7
        self.scale_factor = 0.8

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            for i in range(self.budget):
                a, b, c = np.random.choice(self.budget, 3, replace=False)
                mutant = population[a] + self.scale_factor * (population[b] - population[c])
                crossover = np.random.rand(self.dim) < self.crossover_prob
                trial = np.where(crossover, mutant, population[i])
                if func(trial) < func(population[i]):
                    population[i] = trial
        best_idx = np.argmin([func(ind) for ind in population])
        return population[best_idx]