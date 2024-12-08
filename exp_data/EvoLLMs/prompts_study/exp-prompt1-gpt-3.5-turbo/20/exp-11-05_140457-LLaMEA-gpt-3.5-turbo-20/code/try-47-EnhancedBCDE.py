import numpy as np

class EnhancedBCDE(BCDE):
    def dynamic_mutation(self, idx, F):
        F = np.clip(F, 0.1, 0.9)
        return F * np.random.uniform(0.5, 1.0, self.dim)

    def __call__(self, func):
        population = np.random.uniform(self.lb, self.ub, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])
        F = np.random.uniform(0.5, 1.0, self.budget)

        for _ in range(self.budget):
            idx = np.argsort(fitness)
            best = population[idx[0]]

            F = self.dynamic_mutation(idx, F)
            
            mutant = self.boundary_handling(best + F * (population[idx[1]] - population[idx[2]]))
            trial = np.where(np.random.uniform(0, 1, self.dim) < 0.9, mutant, population[idx[0]])

            trial_fitness = func(trial)
            if trial_fitness < fitness[idx[0]]:
                population[idx[0]] = trial
                fitness[idx[0]] = trial_fitness

        return population[idx[0]]