import numpy as np

class EnhancedDynamicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        diversity = np.mean(np.std(population, axis=0))

        for _ in range(self.budget):
            F = np.random.uniform(0.1, 0.9)
            mutant = self.mutation(population, F, diversity)
            trial = np.where(np.random.uniform(0, 1, self.dim) < 0.5, mutant, population)
            population = np.where(np.apply_along_axis(func, 1, trial) < np.apply_along_axis(func, 1, population), trial, population)
            diversity = np.mean(np.std(population, axis=0))

        return population[np.argmin(np.apply_along_axis(func, 1, population))]

    def mutation(self, population, F, diversity):
        rand1, rand2, rand3 = np.random.randint(0, len(population), 3)
        cauchy_scale = 1.0 / (1.0 + diversity)
        mutant = population[rand1] + F * (population[rand2] - population[rand3]) + np.random.standard_cauchy(self.dim) * cauchy_scale
        return np.clip(mutant, -5.0, 5.0)