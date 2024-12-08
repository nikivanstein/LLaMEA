import numpy as np

class ImprovedHybridPSOQIEA:
    def __init__(self, budget, dim, alpha=1.5):
        self.budget = budget
        self.dim = dim
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.alpha = alpha

    def levy_flight(self, shape):
        beta = 1.5
        sigma = np.power(np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)), 1 / beta)
        u = np.random.normal(0, sigma, shape)
        v = np.random.normal(0, 1, shape)
        step = u / np.power(np.abs(v), 1 / beta)
        return 0.01 * step

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        velocity = np.zeros((self.budget, self.dim))
        p_best = population.copy()
        g_best = population[np.argmin([func(ind) for ind in population])]
        
        for _ in range(self.budget):
            r1, r2 = np.random.rand(self.budget, self.dim), np.random.rand(self.budget, self.dim)
            step = self.levy_flight((self.budget, self.dim))
            velocity = self.w * velocity + self.c1 * r1 * (p_best - population) + self.c2 * r2 * (g_best - population) + self.alpha * step
            population = population + velocity
            population = np.clip(population, -5.0, 5.0)
            p_best = np.where(np.array([func(ind) for ind in population]) < np.array([func(ind) for ind in p_best]), population, p_best)
            g_best = population[np.argmin([func(ind) for ind in population])]
        
        return g_best