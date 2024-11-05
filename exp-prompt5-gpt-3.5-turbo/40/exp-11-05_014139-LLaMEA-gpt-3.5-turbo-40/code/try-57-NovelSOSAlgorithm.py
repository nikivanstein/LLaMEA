import numpy as np

class NovelSOSAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.p = 0.5
        self.alpha = 0.1
        self.beta = 0.9
        self.sigma = 0.1
        self.sigma_min = 0.01
        self.sigma_max = 0.2

    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        for _ in range(self.budget):
            for i in range(self.pop_size):
                idxs = np.random.choice(list(range(self.pop_size)), 2, replace=False)
                x_s1, x_s2 = pop[idxs]
                self.sigma = np.clip(self.sigma * np.exp(0.1 * np.random.randn()), self.sigma_min, self.sigma_max)  # Adaptive Sigma
                symbiont = pop[i] + self.p * (x_s1 - x_s2) + np.random.normal(0, self.sigma, self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.alpha:
                        symbiont[j] = pop[i][j]
                    if np.random.rand() < self.beta:
                        symbiont[j] = symbiont[j] + np.random.uniform(-1, 1) * np.abs(symbiont[j] - pop[i][j])
                symbiont_fitness = func(symbiont)
                if symbiont_fitness < fitness[i]:
                    pop[i] = symbiont
                    fitness[i] = symbiont_fitness
        return pop[np.argmin(fitness)]