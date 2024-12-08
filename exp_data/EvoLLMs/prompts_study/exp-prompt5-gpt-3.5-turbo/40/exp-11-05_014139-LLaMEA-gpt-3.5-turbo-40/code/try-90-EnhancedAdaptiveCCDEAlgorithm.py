import numpy as np

class EnhancedAdaptiveCCDEAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.subpop_size = 5
        self.cr = 0.5
        self.f = 0.5
        self.sigma = 0.1
        self.sigma_min = 0.01
        self.sigma_max = 0.2
        self.cr_min = 0.2
        self.cr_max = 0.9

    def __call__(self, func):
        pops = [np.random.uniform(-5.0, 5.0, (self.subpop_size, self.dim)) for _ in range(self.pop_size)]
        fitness = np.array([[func(ind) for ind in pop] for pop in pops])
        
        for _ in range(self.budget):
            for i in range(self.pop_size):
                pop = pops[i]
                for j in range(self.subpop_size):
                    idxs = np.random.choice(list(range(self.subpop_size)), 3, replace=False)
                    x_r1, x_r2, x_r3 = pop[idxs]
                    self.f = np.random.uniform(0.1, 0.9)  # Adaptive F
                    self.cr = np.random.uniform(0.2, 0.9)  # Adaptive CR
                    self.sigma = np.clip(self.sigma * np.exp(0.1 * np.random.randn()), self.sigma_min, self.sigma_max)  # Adaptive Sigma
                    mutant = pop[j] + self.f * (x_r1 - pop[j]) + self.f * (x_r2 - x_r3) + np.random.normal(0, self.sigma, self.dim)
                    for k in range(self.dim):
                        if np.random.rand() > self.cr:
                            mutant[k] = pop[j][k]
                    mutant_fitness = func(mutant)
                    if mutant_fitness < fitness[i][j]:
                        pops[i][j] = mutant
                        fitness[i][j] = mutant_fitness
        best_subpop_idx = np.unravel_index(np.argmin(fitness), fitness.shape)
        return pops[best_subpop_idx[0]][best_subpop_idx[1]]