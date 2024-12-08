import numpy as np

class FastDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.cr = 0.9

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                r = np.random.randint(0, self.dim)
                mutant = np.copy(self.population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.cr or j == r:
                        mutant[j] = self.population[a][j] + 0.5 * (self.population[b][j] - self.population[c][j])
                trial = np.where(np.logical_or(mutant < -5.0, mutant > 5.0), self.population[i], mutant)
                if func(trial) < func(self.population[i]):
                    self.population[i] = trial

        final_fitness = [func(individual) for individual in self.population]
        best_idx = np.argmin(final_fitness)
        best_individual = self.population[best_idx]

        return best_individual