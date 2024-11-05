import numpy as np

class AdaptiveEnhancedDynamicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        F_min, F_max = 0.1, 0.9
        CR = 0.9

        for _ in range(self.budget):
            F = F_min + (F_max - F_min) * np.random.rand()
            diversity = np.std(population)
            scale_factor = 1 / (1 + np.exp(-diversity))  # Adaptive Cauchy scale factor
            rand1, rand2, rand3 = np.random.randint(0, len(population), 3)
            mutant = population[rand1] + F * (population[rand2] - population[rand3]) + np.random.standard_cauchy(self.dim) * scale_factor
            trial = np.where(np.random.rand(self.dim) < CR, mutant, population[rand1])
            population[rand1] = np.clip(trial, -5.0, 5.0)

        fitness = [func(ind) for ind in population]
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        return best_solution