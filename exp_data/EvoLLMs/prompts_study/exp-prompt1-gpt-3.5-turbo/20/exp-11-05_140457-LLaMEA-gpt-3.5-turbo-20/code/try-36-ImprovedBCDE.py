import numpy as np

class ImprovedBCDE(BCDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def dynamic_mutation(self, x, f):
        scale_factor = 0.8 + 0.2 * np.tanh(f)
        return np.clip(x, self.lb, self.ub)

    def __call__(self, func):
        population = np.random.uniform(self.lb, self.ub, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])

        for _ in range(self.budget):
            idx = np.argsort(fitness)
            best = population[idx[0]]

            f_factor = np.mean(fitness) / fitness[idx[0]]
            mutant = self.dynamic_mutation(best + f_factor * (population[idx[1]] - population[idx[2]]), f_factor)
            trial = np.where(np.random.uniform(0, 1, self.dim) < 0.9, mutant, population[idx[0]])

            trial_fitness = func(trial)
            if trial_fitness < fitness[idx[0]]:
                population[idx[0]] = trial
                fitness[idx[0]] = trial_fitness

        return population[idx[0]]