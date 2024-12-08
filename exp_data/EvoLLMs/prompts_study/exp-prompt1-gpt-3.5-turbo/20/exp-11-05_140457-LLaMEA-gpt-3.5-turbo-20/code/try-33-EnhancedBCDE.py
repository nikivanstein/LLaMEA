import numpy as np

class EnhancedBCDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.F = 0.8
        self.CR = 0.9

    def boundary_handling(self, x):
        return np.clip(x, self.lb, self.ub)

    def __call__(self, func):
        population = np.random.uniform(self.lb, self.ub, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])
        F = self.F
        CR = self.CR

        for _ in range(self.budget):
            idx = np.argsort(fitness)
            best = population[idx[0]]

            F = max(0.5, min(1.0, F + 0.05 * np.random.randn()))
            CR = max(0.1, min(0.9, CR + 0.05 * np.random.randn()))

            mutant = self.boundary_handling(best + F * (population[idx[1]] - population[idx[2]]))
            trial = np.where(np.random.uniform(0, 1, self.dim) < CR, mutant, population[idx[0]])

            trial_fitness = func(trial)
            if trial_fitness < fitness[idx[0]]:
                population[idx[0]] = trial
                fitness[idx[0]] = trial_fitness

        return population[idx[0]]