import numpy as np

class AdaptiveInertiaHybridPSOQIEA(HybridPSOQIEA):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.w_min = 0.4
        self.w_max = 0.9

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        velocity = np.zeros((self.budget, self.dim))
        p_best = population.copy()
        g_best = population[np.argmin([func(ind) for ind in population])
        inertia_weight = self.w_max

        for _ in range(self.budget):
            r1, r2 = np.random.rand(self.budget, self.dim), np.random.rand(self.budget, self.dim)
            velocity = inertia_weight * velocity + self.c1 * r1 * (p_best - population) + self.c2 * r2 * (g_best - population)
            population = population + velocity
            population = np.clip(population, -5.0, 5.0)
            p_best = np.where(np.array([func(ind) for ind in population]) < np.array([func(ind) for ind in p_best]), population, p_best)
            g_best = population[np.argmin([func(ind) for ind in population])]
            inertia_weight = self.w_max - ((_ + 1) / self.budget) * (self.w_max - self.w_min)

        return g_best