import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.w = 0.5  # Initial inertia weight
        self.c1 = 2.0  # Cognitive parameter
        self.c2 = 2.0  # Social parameter
        self.v_max = 0.2 * (5.0 - (-5.0))  # Maximum velocity
        self.population = np.random.uniform(-5.0, 5.0, (dim,))
        self.p_best = self.population.copy()
        self.p_best_fitness = float('inf')
        self.g_best = self.population[np.argmin(func(self.population))]
        self.g_best_fitness = func(self.g_best)

    def __call__(self, func):
        for _ in range(self.budget):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            velocity = self.w * velocity + self.c1 * r1 * (self.p_best - self.population) + self.c2 * r2 * (self.g_best - self.population)
            velocity = np.clip(velocity, -self.v_max, self.v_max)
            self.population += velocity
            self.population = np.clip(self.population, -5.0, 5.0)
            fitness = func(self.population)
            if fitness < self.p_best_fitness:
                self.p_best = self.population
                self.p_best_fitness = fitness
            if fitness < self.g_best_fitness:
                self.g_best = self.population
                self.g_best_fitness = fitness
            self.w = 0.5 + 0.5 * np.exp(-_ / self.budget)  # Dynamic inertia weight adjustment
        return self.g_best