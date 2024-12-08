import numpy as np

class ImprovedBCDE(BCDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def dynamic_mutation(self, x, f):
        return np.clip(x + f * np.random.normal(0, 1, self.dim), self.lb, self.ub)

    def __call__(self, func):
        population = np.random.uniform(self.lb, self.ub, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])

        for _ in range(self.budget):
            idx = np.argsort(fitness)
            best = population[idx[0]]

            f = max(0.1, 0.8 - 0.7 * _ / self.budget)  # Dynamic mutation factor
            mutant = self.boundary_handling(best + f * (population[idx[1]] - population[idx[2]]))
            trial = np.where(np.random.uniform(0, 1, self.dim) < 0.9, mutant, population[idx[0]])

            trial = self.dynamic_mutation(trial, f)  # Introduce dynamic mutation

            trial_fitness = func(trial)
            if trial_fitness < fitness[idx[0]]:
                population[idx[0]] = trial
                fitness[idx[0]] = trial_fitness

        return population[idx[0]]