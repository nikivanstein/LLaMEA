import numpy as np

class AdaptiveDEAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.cr = 0.5
        self.f_min = 0.2
        self.f_max = 0.8
        self.sigma = 0.1

    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_current = self.f_max

        for _ in range(self.budget):
            for i in range(self.pop_size):
                idxs = np.random.choice(list(range(self.pop_size)), 3, replace=False)
                x_r1, x_r2, x_r3 = pop[idxs]
                mutant = pop[i] + f_current * (x_r1 - pop[i]) + f_current * (x_r2 - x_r3) + np.random.normal(0, self.sigma, self.dim)
                for j in range(self.dim):
                    if np.random.rand() > self.cr:
                        mutant[j] = pop[i][j]
                mutant_fitness = func(mutant)
                if mutant_fitness < fitness[i]:
                    pop[i] = mutant
                    fitness[i] = mutant_fitness
                    f_current = min(self.f_max, f_current * 1.2)
                else:
                    f_current = max(self.f_min, f_current * 0.8)
        return pop[np.argmin(fitness)]