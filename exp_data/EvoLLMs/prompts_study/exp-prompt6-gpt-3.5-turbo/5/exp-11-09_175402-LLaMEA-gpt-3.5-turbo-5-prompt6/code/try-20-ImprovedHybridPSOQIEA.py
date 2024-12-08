import numpy as np

class ImprovedHybridPSOQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.min_w = 0.4
        self.max_w = 0.9
        self.c1 = 1.5
        self.c2 = 1.5

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        velocity = np.zeros((self.budget, self.dim))
        p_best = population.copy()
        g_best = population[np.argmin([func(ind) for ind in population])]
        iterations = 0
        
        while iterations < self.budget:
            w = self.max_w - iterations * ((self.max_w - self.min_w) / self.budget)
            r1, r2 = np.random.rand(self.budget, self.dim), np.random.rand(self.budget, self.dim)
            velocity = w * velocity + self.c1 * r1 * (p_best - population) + self.c2 * r2 * (g_best - population)
            population = population + velocity
            population = np.clip(population, -5.0, 5.0)
            p_best = np.where(np.array([func(ind) for ind in population]) < np.array([func(ind) for ind in p_best]), population, p_best)
            g_best = population[np.argmin([func(ind) for ind in population])]
            iterations += 1
        
        return g_best