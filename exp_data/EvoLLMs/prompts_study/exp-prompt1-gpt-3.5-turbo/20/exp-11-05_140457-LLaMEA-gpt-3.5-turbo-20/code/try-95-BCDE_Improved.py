import numpy as np

class BCDE_Improved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0

    def boundary_handling(self, x):
        return np.clip(x, self.lb, self.ub)

    def opposition_based(self, x):
        return self.lb + self.ub - x

    def __call__(self, func):
        population = np.random.uniform(self.lb, self.ub, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])

        for _ in range(self.budget):
            idx = np.argsort(fitness)
            best = population[idx[0]]

            mutant = self.boundary_handling(best + 0.8 * (population[idx[1]] - population[idx[2]]))
            trial = np.where(np.random.uniform(0, 1, self.dim) < 0.9, mutant, population[idx[0]])

            # Integrate opposition-based learning
            trial_opposite = self.opposition_based(trial)
            trial_opposite_fitness = func(trial_opposite)
            if trial_opposite_fitness < fitness[idx[0]]:
                population[idx[0]] = trial_opposite
                fitness[idx[0]] = trial_opposite_fitness
            else:
                trial_fitness = func(trial)
                if trial_fitness < fitness[idx[0]]:
                    population[idx[0]] = trial
                    fitness[idx[0]] = trial_fitness

        return population[idx[0]]