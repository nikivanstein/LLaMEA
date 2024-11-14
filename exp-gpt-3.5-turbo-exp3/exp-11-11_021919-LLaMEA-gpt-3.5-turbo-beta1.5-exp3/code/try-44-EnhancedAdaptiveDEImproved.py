import numpy as np

class EnhancedAdaptiveDEImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.F_min, self.F_max = 0.2, 0.8
        self.CR_min, self.CR_max = 0.2, 0.8

    def dynamic_population_size(self, t):
        return int(10 + 40 * (1 - np.exp(-t / 800)))

    def clip_to_bounds(self, x):
        return np.clip(x, -5.0, 5.0)

    def optimize(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)

        for _ in range(self.budget):
            population_size = self.dynamic_population_size(_)
            population = np.array([np.random.uniform(-5.0, 5.0, self.dim) for _ in range(population_size)])

            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                F = np.random.uniform(self.F_min, self.F_max)
                CR = np.random.uniform(self.CR_min, self.CR_max)

                mutant = self.clip_to_bounds(a + F * (b - c))
                crossover_mask = np.random.rand(self.dim) < CR
                trial = np.where(crossover_mask, mutant, population[i])

                trial_fitness = func(trial)
                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness

        return best_solution