import numpy as np

class DynamicInertiaHybridPSOQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.c1 = 1.5
        self.c2 = 1.5

    def __call__(self, func):
        w_min, w_max = 0.4, 0.9
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        velocity = np.zeros((self.budget, self.dim))
        p_best = population.copy()
        g_best = population[np.argmin([func(ind) for ind in population])]
        w = w_max

        for _ in range(self.budget):
            r1, r2 = np.random.rand(self.budget, self.dim), np.random.rand(self.budget, self.dim)
            velocity = w * velocity + self.c1 * r1 * (p_best - population) + self.c2 * r2 * (g_best - population)
            population = population + velocity
            population = np.clip(population, -5.0, 5.0)
            p_best = np.where(np.array([func(ind) for ind in population]) < np.array([func(ind) for ind in p_best]), population, p_best)
            new_g_best = population[np.argmin([func(ind) for ind in population])]
            if func(new_g_best) < func(g_best):
                g_best = new_g_best
            w = w_max - (_ / self.budget) * (w_max - w_min)

        return g_best