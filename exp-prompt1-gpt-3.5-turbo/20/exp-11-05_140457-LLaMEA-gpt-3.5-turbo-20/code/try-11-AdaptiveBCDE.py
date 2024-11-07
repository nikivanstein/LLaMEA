import numpy as np

class AdaptiveBCDE(BCDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.scale_factor = 0.8

    def adaptive_mutation_scaling(self, idx, n):
        return self.scale_factor * np.exp(-idx / n)

    def __call__(self, func):
        population = np.random.uniform(self.lb, self.ub, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])

        for idx in range(self.budget):
            sort_idx = np.argsort(fitness)
            best = population[sort_idx[0]]

            scale = self.adaptive_mutation_scaling(idx, self.budget)
            mutant = self.boundary_handling(best + scale * (population[sort_idx[1]] - population[sort_idx[2]]))
            trial = np.where(np.random.uniform(0, 1, self.dim) < 0.9, mutant, population[sort_idx[0]])

            trial_fitness = func(trial)
            if trial_fitness < fitness[sort_idx[0]]:
                population[sort_idx[0]] = trial
                fitness[sort_idx[0]] = trial_fitness

        return population[sort_idx[0]]