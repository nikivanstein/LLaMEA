import numpy as np

class EnhancedBCDE(BCDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def dynamic_mutation(self, generation):
        return 0.5 + 0.5 * np.exp(-4 * generation / self.budget)

    def __call__(self, func):
        population = np.random.uniform(self.lb, self.ub, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])

        for gen in range(self.budget):
            idx = np.argsort(fitness)
            best = population[idx[0]]

            mutation_factor = self.dynamic_mutation(gen)
            mutant = self.boundary_handling(best + mutation_factor * (population[idx[1]] - population[idx[2]]))
            trial = np.where(np.random.uniform(0, 1, self.dim) < 0.9, mutant, population[idx[0]])

            trial_fitness = func(trial)
            if trial_fitness < fitness[idx[0]]:
                population[idx[0]] = trial
                fitness[idx[0]] = trial_fitness

        return population[idx[0]]