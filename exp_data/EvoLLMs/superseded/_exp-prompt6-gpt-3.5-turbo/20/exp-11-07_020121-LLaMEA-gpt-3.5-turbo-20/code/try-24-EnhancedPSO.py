import numpy as np

class EnhancedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.w = 0.9  # Initial inertia weight
        self.c1 = 1.496
        self.c2 = 1.496
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.velocities = np.zeros((budget, dim))

    def __call__(self, func):
        for _ in range(self.budget):
            fitness_values = func(self.population)
            g_best_idx = np.argmin(fitness_values)
            g_best = self.population[g_best_idx]
            
            r1, r2 = np.random.rand(), np.random.rand()
            self.velocities = self.w * self.velocities + self.c1 * r1 * (self.population - self.population) + self.c2 * r2 * (g_best - self.population)
            self.population = np.clip(self.population + self.velocities, -5.0, 5.0)
            
        return g_best