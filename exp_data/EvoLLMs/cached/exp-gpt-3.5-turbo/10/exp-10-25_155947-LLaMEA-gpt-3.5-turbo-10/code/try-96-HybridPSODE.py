import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.c1 = 1.5  # Cognitive component weight
        self.c2 = 1.5  # Social component weight
        self.w = 0.7   # Inertia weight
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))

        population = initialize_population()
        velocities = np.zeros((self.population_size, self.dim))
        personal_best = population.copy()
        global_best = population[np.argmin([func(individual) for individual in population])]

        for _ in range(self.budget):
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)

            velocities = self.w * velocities + self.c1 * r1 * (personal_best - population) + self.c2 * r2 * (global_best - population)
            population = population + velocities
            population = np.clip(population, self.lb, self.ub)

            for i in range(self.population_size):
                if func(population[i]) < func(personal_best[i]):
                    personal_best[i] = population[i]
                    if func(personal_best[i]) < func(global_best):
                        global_best = personal_best[i]

        return global_best