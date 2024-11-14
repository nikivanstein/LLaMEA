import numpy as np

class OptimizedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.w = 0.9  # Initial inertia weight
        self.c1 = 1.496
        self.c2 = 1.496
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.velocities = np.zeros((budget, dim))

    def __call__(self, func):
        func_pop = np.apply_along_axis(func, 1, self.population)  # Evaluate function for entire population
        g_best_idx = np.argmin(func_pop)
        g_best = self.population[g_best_idx]

        for _ in range(self.budget):
            r1, r2 = np.random.rand(), np.random.rand()
            cognitive = self.c1 * r1 * (self.population - self.population)
            social = self.c2 * r2 * (g_best - self.population)
            self.velocities = self.w * self.velocities + cognitive + social
            self.population = np.clip(self.population + self.velocities, -5.0, 5.0)

            new_func_pop = np.apply_along_axis(func, 1, self.population)  # Evaluate new population
            new_g_best_idx = np.argmin(new_func_pop)
            g_best, g_best_idx, func_pop = np.where(new_func_pop < func_pop, (self.population[new_g_best_idx], new_g_best_idx, new_func_pop), (g_best, g_best_idx, func_pop))

        return g_best