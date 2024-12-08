import numpy as np

class ImprovedHybridPSOQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5

    def chaotic_initialization(self, size):
        population = []
        for _ in range(size):
            x = np.random.uniform(-5.0, 5.0, self.dim)
            for _ in range(100):
                x = np.sin(x)
            population.append(x)
        return np.array(population)

    def __call__(self, func):
        population = self.chaotic_initialization(self.budget)
        velocity = np.zeros((self.budget, self.dim))
        p_best = population.copy()
        g_best = population[np.argmin([func(ind) for ind in population])]
        
        for _ in range(self.budget):
            r1, r2 = np.random.rand(self.budget, self.dim), np.random.rand(self.budget, self.dim)
            velocity = self.w * velocity + self.c1 * r1 * (p_best - population) + self.c2 * r2 * (g_best - population)
            population = population + velocity
            population = np.clip(population, -5.0, 5.0)
            p_best = np.where(np.array([func(ind) for ind in population]) < np.array([func(ind) for ind in p_best]), population, p_best)
            g_best = population[np.argmin([func(ind) for ind in population])]
        
        return g_best