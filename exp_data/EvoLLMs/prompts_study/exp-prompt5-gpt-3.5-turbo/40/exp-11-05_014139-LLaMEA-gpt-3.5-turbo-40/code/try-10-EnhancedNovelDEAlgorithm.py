import numpy as np

class EnhancedNovelDEAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.sigma = 0.1
        self.min_f = 0.5
        self.max_f = 1.0
        self.min_cr = 0.1
        self.max_cr = 0.9

    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        f = self.min_f + np.random.rand() * (self.max_f - self.min_f)
        cr = self.min_cr + np.random.rand() * (self.max_cr - self.min_cr)

        for _ in range(self.budget):
            for i in range(self.pop_size):
                idxs = np.random.choice(list(range(self.pop_size)), 3, replace=False)
                x_r1, x_r2, x_r3 = pop[idxs]
                mutant = pop[i] + f * (x_r1 - pop[i]) + f * (x_r2 - x_r3) + np.random.normal(0, self.sigma, self.dim)
                for j in range(self.dim):
                    if np.random.rand() > cr:
                        mutant[j] = pop[i][j]
                mutant_fitness = func(mutant)
                if mutant_fitness < fitness[i]:
                    pop[i] = mutant
                    fitness[i] = mutant_fitness
                    if mutant_fitness < fitness[best_idx]:
                        best_idx = i
                        f = max(self.min_f, min(self.max_f, f + np.random.normal(0, 0.1)))
                        cr = max(self.min_cr, min(self.max_cr, cr + np.random.normal(0, 0.1)))
        return pop[best_idx]