import numpy as np

class SelfAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.cr = 0.5
        self.F_min = 0.2
        self.F_max = 0.8

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        F = np.random.uniform(self.F_min, self.F_max, self.budget)
        
        for _ in range(self.budget):
            for i in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                d = np.random.randint(0, self.dim)
                trial = np.where(np.random.rand(self.dim) < self.cr, a + F[i] * (b - c), population[i])
                trial[d] = best[d]
                if func(trial) < fitness[i]:
                    population[i] = trial
                    fitness[i] = func(trial)
                    if fitness[i] < func(best):
                        best = trial

        return best