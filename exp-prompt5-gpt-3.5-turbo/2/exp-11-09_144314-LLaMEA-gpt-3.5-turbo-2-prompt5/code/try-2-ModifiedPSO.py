import numpy as np

class ModifiedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight = 0.5 + np.random.rand() * 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.v_max = (5.0 - (-5.0)) * 0.1
        self.v_min = -self.v_max
        self.global_best_position = np.random.uniform(-5.0, 5.0, dim)
        self.global_best_fitness = float('inf')
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, dim))
        self.fitness_values = np.apply_along_axis(func, 1, self.population)

    def __call__(self, func):
        while self.budget > 0:
            for i in range(self.population_size):
                fitness_val = func(self.population[i])
                if fitness_val < self.fitness_values[i]:
                    self.fitness_values[i] = fitness_val
                    if fitness_val < self.global_best_fitness:
                        self.global_best_fitness = fitness_val
                        self.global_best_position = self.population[i]
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)
                self.population[i] = self.population[i] + self.inertia_weight * self.population[i] + self.c1 * r1 * (self.global_best_position - self.population[i]) + self.c2 * r2 * (self.global_best_position - self.population[i])
                self.population[i] = np.clip(self.population[i], -5.0, 5.0)
                self.budget -= 1
        return self.global_best_position