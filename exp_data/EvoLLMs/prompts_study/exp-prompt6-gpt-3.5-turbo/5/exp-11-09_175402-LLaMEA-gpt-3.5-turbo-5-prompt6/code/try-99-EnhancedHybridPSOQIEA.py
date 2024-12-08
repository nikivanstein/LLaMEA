import numpy as np

class EnhancedHybridPSOQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.f = 0.5
        self.cr = 0.9

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        velocity = np.zeros((self.budget, self.dim))
        p_best = population.copy()
        g_best = population[np.argmin([func(ind) for ind in population])
        
        for _ in range(self.budget):
            r1, r2 = np.random.rand(self.budget, self.dim), np.random.rand(self.budget, self.dim)
            mutation = p_best + self.f * (p_best - population) + self.cr * (population - population[np.argsort([func(ind) for ind in population])[1]])
            velocity = self.w * velocity + self.c1 * r1 * (p_best - population) + self.c2 * r2 * (g_best - population)
            population = population + velocity
            population = np.clip(population, -5.0, 5.0)
            p_best = np.where(np.array([func(ind) for ind in population]) < np.array([func(ind) for ind in p_best]), population, p_best)
            g_best = population[np.argmin([func(ind) for ind in population])]
        
        return g_best