import numpy as np

class ImprovedBCDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.f_scale = 0.8

    def boundary_handling(self, x):
        return np.clip(x, self.lb, self.ub)

    def adaptive_scale(self, scale):
        return max(0.1, min(0.9, scale + np.random.normal(0.5, 0.1)))

    def __call__(self, func):
        population = np.random.uniform(self.lb, self.ub, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])

        for _ in range(self.budget):
            idx = np.argsort(fitness)
            best = population[idx[0]]

            mutant = self.boundary_handling(best + self.f_scale * (population[idx[1]] - population[idx[2]]))
            trial = np.where(np.random.uniform(0, 1, self.dim) < 0.9, mutant, population[idx[0]])

            trial_fitness = func(trial)
            if trial_fitness < fitness[idx[0]]:
                population[idx[0]] = trial
                fitness[idx[0]] = trial_fitness
                self.f_scale = self.adaptive_scale(self.f_scale)

        return population[idx[0]]