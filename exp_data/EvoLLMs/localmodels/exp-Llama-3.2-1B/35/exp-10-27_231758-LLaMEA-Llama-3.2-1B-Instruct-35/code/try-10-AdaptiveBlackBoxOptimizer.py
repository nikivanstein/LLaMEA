import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.population = None

    def __call__(self, func, population=1):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

        # Select the next individual based on probability
        self.population = population
        if self.population == 1:
            return func(self.func_values)
        else:
            return np.random.choice([func, self.func_values], size=self.population, p=[1-self.population/2, self.population/2])

    def fit(self, func, population_size=100, population_size_decrease=0.99, mutation_rate=0.01, crossover_rate=0.5):
        for _ in range(1000):
            new_population = self.__call__(func, population_size)
            self.population = population_size
            if np.mean(np.abs(new_population - self.func_values)) < 1e-6:
                break
            self.population = population_size_decrease * self.population + (1 - population_size_decrease) * population_size
            self.func_values = new_population