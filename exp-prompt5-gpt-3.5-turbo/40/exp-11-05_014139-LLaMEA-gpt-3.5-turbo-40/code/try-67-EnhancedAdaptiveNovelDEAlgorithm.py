import numpy as np

class EnhancedAdaptiveNovelDEAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.sigma = 0.1
        self.sigma_min = 0.01
        self.sigma_max = 0.2
        self.sigma_factor = 0.1
        self.cr_min = 0.2
        self.cr_max = 0.9
        self.cr_factor = 0.1
        self.f_min = 0.1
        self.f_max = 0.9
        self.f_factor = 0.1

    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        
        for _ in range(self.budget):
            for i in range(self.pop_size):
                idxs = np.random.choice(list(range(self.pop_size)), 3, replace=False)
                x_r1, x_r2, x_r3 = pop[idxs]
                self.f = np.clip(np.random.normal(self.f, self.f_factor), self.f_min, self.f_max)  # Adaptive F
                self.cr = np.clip(np.random.normal(self.cr, self.cr_factor), self.cr_min, self.cr_max)  # Adaptive CR
                self.sigma = np.clip(self.sigma * np.exp(self.sigma_factor * np.random.randn()), self.sigma_min, self.sigma_max)  # Adaptive Sigma
                mutant = pop[i] + self.f * (x_r1 - pop[i]) + self.f * (x_r2 - x_r3) + np.random.normal(0, self.sigma, self.dim)
                for j in range(self.dim):
                    if np.random.rand() > self.cr:
                        mutant[j] = pop[i][j]
                mutant_fitness = func(mutant)
                if mutant_fitness < fitness[i]:
                    pop[i] = mutant
                    fitness[i] = mutant_fitness
        return pop[np.argmin(fitness)]