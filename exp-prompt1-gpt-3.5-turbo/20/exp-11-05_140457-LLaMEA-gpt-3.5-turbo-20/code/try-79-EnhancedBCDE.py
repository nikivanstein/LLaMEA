import numpy as np

class EnhancedBCDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0

    def boundary_handling(self, x):
        return np.clip(x, self.lb, self.ub)

    def dynamic_mutation(self, x_best, x_r1, x_r2, f):
        return np.clip(x_best + np.clip(f, 0.1, 0.9) * (x_r1 - x_r2), self.lb, self.ub)

    def __call__(self, func):
        population = np.random.uniform(self.lb, self.ub, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])
        f = 0.5

        for _ in range(self.budget):
            idx = np.argsort(fitness)
            best = population[idx[0]]

            mutant = self.boundary_handling(self.dynamic_mutation(best, population[idx[1]], population[idx[2]], f))
            trial = np.where(np.random.uniform(0, 1, self.dim) < 0.9, mutant, population[idx[0]])

            trial_fitness = func(trial)
            if trial_fitness < fitness[idx[0]]:
                population[idx[0]] = trial
                fitness[idx[0]] = trial_fitness
                f = max(0.1, f * 0.9)

        return population[idx[0]]