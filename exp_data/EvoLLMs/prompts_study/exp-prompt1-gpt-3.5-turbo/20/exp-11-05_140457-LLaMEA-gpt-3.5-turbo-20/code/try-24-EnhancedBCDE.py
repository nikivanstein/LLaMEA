import numpy as np

class EnhancedBCDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.f = 0.8
        self.cr = 0.9

    def boundary_handling(self, x):
        return np.clip(x, self.lb, self.ub)

    def adaptive_mutation(self, population, fitness, idx):
        f = np.clip(0.1 + 0.9 * np.exp(-3.0 * np.arange(0, 1, 1 / self.budget)), 0, 1)
        for i in range(self.budget):
            mutant = self.boundary_handling(population[idx[i]] + f[i] * (population[idx[i+1]] - population[idx[i+2]]))
            trial = np.where(np.random.uniform(0, 1, self.dim) < self.cr, mutant, population[idx[i]])
            trial_fitness = func(trial)
            if trial_fitness < fitness[idx[i]]:
                population[idx[i]] = trial
                fitness[idx[i]] = trial_fitness
        return population, fitness

    def __call__(self, func):
        population = np.random.uniform(self.lb, self.ub, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])

        for _ in range(self.budget):
            idx = np.argsort(fitness)
            population, fitness = self.adaptive_mutation(population, fitness, idx)

        return population[idx[0]]