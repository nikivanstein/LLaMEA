import numpy as np

class BCDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.F = 0.8

    def boundary_handling(self, x):
        return np.clip(x, self.lb, self.ub)

    def __call__(self, func):
        population = np.random.uniform(self.lb, self.ub, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])
        scaling_factor = np.full(self.budget, self.F)

        for _ in range(self.budget):
            idx = np.argsort(fitness)
            best = population[idx[0]]

            mutant = self.boundary_handling(best + scaling_factor[idx[0]] * (population[idx[1]] - population[idx[2]]))
            trial = np.where(np.random.uniform(0, 1, self.dim) < 0.9, mutant, population[idx[0]])

            trial_fitness = func(trial)
            if trial_fitness < fitness[idx[0]]:
                population[idx[0]] = trial
                fitness[idx[0]] = trial_fitness
                if trial_fitness < fitness[idx[1]]:
                    scaling_factor[idx[0]] *= 1.2
                else:
                    scaling_factor[idx[0]] *= 0.9

        return population[idx[0]]