import numpy as np

class ProbabilisticLineChangeOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        for _ in range(self.budget):
            for i in range(self.population_size):
                x, a, b, c = population[i], population[np.random.randint(self.population_size)], population[np.random.randint(self.population_size)], population[np.random.randint(self.population_size)]
                if np.random.rand() < 0.2:
                    self.f = np.clip(np.random.normal(self.f, 0.1), self.min_f, self.max_f)  # Probabilistic mutation rate change
                mutant = np.clip(a + self.f * (b - c), -5.0, 5.0)
                trial = np.where(np.random.rand(self.dim) <= self.cr, mutant, x)
                if func(trial) < func(x):
                    population[i] = trial.copy()
        return population