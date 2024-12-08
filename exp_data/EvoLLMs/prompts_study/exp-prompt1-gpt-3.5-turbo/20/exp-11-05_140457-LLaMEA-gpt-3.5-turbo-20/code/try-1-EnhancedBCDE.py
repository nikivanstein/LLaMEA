import numpy as np

class EnhancedBCDE(BCDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def dynamic_mutation(self, best, ind1, ind2, f=0.8):
        scale_factor = np.clip(0.5 + np.random.normal(0, 0.1), 0, 1)
        return self.boundary_handling(best + scale_factor * (ind1 - ind2))

    def __call__(self, func):
        population = np.random.uniform(self.lb, self.ub, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])

        for _ in range(self.budget):
            idx = np.argsort(fitness)
            best = population[idx[0]]

            mutant = self.dynamic_mutation(best, population[idx[1]], population[idx[2]])
            trial = np.where(np.random.uniform(0, 1, self.dim) < 0.9, mutant, population[idx[0]])

            trial_fitness = func(trial)
            if trial_fitness < fitness[idx[0]]:
                population[idx[0]] = trial
                fitness[idx[0]] = trial_fitness

        return population[idx[0]]