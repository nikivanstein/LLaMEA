import numpy as np

class ImprovedBCDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.cr = 0.9
        self.f = 0.8

    def boundary_handling(self, x):
        return np.clip(x, self.lb, self.ub)

    def adaptive_mutation(self, population, idx, scaling_factor):
        return self.boundary_handling(population[idx[0]] + scaling_factor * (population[idx[1]] - population[idx[2]]))

    def __call__(self, func):
        population = np.random.uniform(self.lb, self.ub, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])

        for _ in range(self.budget):
            idx = np.argsort(fitness)
            best = population[idx[0]]

            mutant = self.adaptive_mutation(population, idx, self.f)
            trial = np.where(np.random.uniform(0, 1, self.dim) < self.cr, mutant, population[idx[0]])

            trial_fitness = func(trial)
            if trial_fitness < fitness[idx[0]]:
                population[idx[0]] = trial
                fitness[idx[0]] = trial_fitness

        return population[idx[0]]