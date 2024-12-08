import numpy as np

class EnhancedBCDE(BCDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def gaussian_mutation(self, x, scale=0.1):
        return self.boundary_handling(x + np.random.normal(0, scale, self.dim))

    def __call__(self, func):
        population = np.random.uniform(self.lb, self.ub, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])

        for _ in range(self.budget):
            idx = np.argsort(fitness)
            best = population[idx[0]]

            mutant = self.boundary_handling(best + 0.8 * (population[idx[1]] - population[idx[2]]))
            trial = np.where(np.random.uniform(0, 1, self.dim) < 0.9, mutant, population[idx[0]])
            trial = self.gaussian_mutation(trial)  # Adding Gaussian mutation for better exploration

            trial_fitness = func(trial)
            if trial_fitness < fitness[idx[0]]:
                population[idx[0]] = trial
                fitness[idx[0]] = trial_fitness

        return population[idx[0]]