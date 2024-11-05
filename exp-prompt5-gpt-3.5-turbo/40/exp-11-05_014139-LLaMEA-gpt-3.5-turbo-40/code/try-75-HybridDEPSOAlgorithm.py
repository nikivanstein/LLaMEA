import numpy as np

class HybridDEPSOAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.w = 0.5
        self.w_min = 0.1
        self.w_max = 0.9
        self.sigma_min = 0.01
        self.sigma_max = 0.2
        self.cr_min = 0.2
        self.cr_max = 0.9

    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        
        for _ in range(self.budget):
            for i in range(self.pop_size):
                idxs = np.random.choice(list(range(self.pop_size)), 3, replace=False)
                x_r1, x_r2, x_r3 = pop[idxs]
                self.w = np.clip(self.w * np.exp(-0.1 * np.random.randn()), self.w_min, self.w_max)  # Adaptive Inertia Weight
                self.c1 = np.random.uniform(0, 2.0)  # Random cognitive component
                self.c2 = np.random.uniform(0, 2.0)  # Random social component

                mutant = pop[i] + self.c1 * np.random.uniform(0, 1, self.dim) * (x_r1 - pop[i]) + self.c2 * np.random.uniform(0, 1, self.dim) * (x_r2 - x_r3)
                for j in range(self.dim):
                    if np.random.rand() > self.cr:
                        mutant[j] = pop[i][j]
                mutant_fitness = func(mutant)
                if mutant_fitness < fitness[i]:
                    pop[i] = mutant
                    fitness[i] = mutant_fitness
        return pop[np.argmin(fitness)]