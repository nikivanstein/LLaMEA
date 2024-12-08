import numpy as np

class SelfAdaptiveDE:
    def __init__(self, budget, dim, population_size=30, scaling_factor=0.5, crossover_rate=0.9, min_scaling_factor=0.2, max_scaling_factor=0.8, min_crossover_rate=0.6, max_crossover_rate=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.scaling_factor = scaling_factor
        self.crossover_rate = crossover_rate
        self.min_scaling_factor = min_scaling_factor
        self.max_scaling_factor = max_scaling_factor
        self.min_crossover_rate = min_crossover_rate
        self.max_crossover_rate = max_crossover_rate

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        population = initialize_population()
        fitness = np.array([func(individual) for individual in population])
        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.scaling_factor * (b - c), -5.0, 5.0)
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover_mask, mutant, population[i])
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Update scaling factor and crossover rate
            self.scaling_factor = np.clip(self.scaling_factor + np.random.normal(0, 0.1), self.min_scaling_factor, self.max_scaling_factor)
            self.crossover_rate = np.clip(self.crossover_rate + np.random.normal(0, 0.1), self.min_crossover_rate, self.max_crossover_rate)

        best_idx = np.argmin(fitness)
        return population[best_idx]