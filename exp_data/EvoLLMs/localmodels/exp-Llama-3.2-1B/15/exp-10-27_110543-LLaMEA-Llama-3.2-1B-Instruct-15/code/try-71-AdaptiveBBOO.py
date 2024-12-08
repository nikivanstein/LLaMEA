import numpy as np
from scipy.optimize import minimize

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = []
        self.algorithm = 'EvolutionStrategies'

    def __call__(self, func):
        def eval_func(x):
            return func(x)

        def evaluate_budget(func, x, budget):
            if budget <= 0:
                raise ValueError("Budget cannot be zero or negative")
            return np.sum([eval_func(x + np.random.normal(0, 1, size=self.dim)) for _ in range(budget)])

        for _ in range(self.budget):
            for i in range(self.population_size):
                fitness = evaluate_budget(eval_func, self.population[i], self.budget)
                self.fitnesses[i] = fitness
                self.population_history.append(self.population[i])

        # Select the fittest individuals
        self.population = self.population[np.argsort(self.fitnesses, axis=0)]
        self.fitnesses = self.fitnesses[np.argsort(self.fitnesses, axis=0)]

        # Evolve the population
        for _ in range(100):
            if self.algorithm == 'EvolutionStrategies':
                # Refine the strategy by changing the individual lines
                for i in range(self.population_size):
                    if self.fitnesses[i] < 0.75 * self.fitnesses[i]:
                        self.population[i] = np.random.uniform(-5.0, 0.0) + np.random.normal(0.0, 1.0, size=self.dim)
                    elif self.fitnesses[i] > 0.75 * self.fitnesses[i]:
                        self.population[i] = np.random.uniform(0.0, 5.0) + np.random.normal(0.0, 1.0, size=self.dim)
                    else:
                        self.population[i] = self.population[i]

            elif self.algorithm == 'AdaptiveBBOO':
                # Update the strategy by changing the individual lines
                for i in range(self.population_size):
                    if np.random.rand() < 0.15:
                        self.population[i] = np.random.uniform(-5.0, 0.0) + np.random.normal(0.0, 1.0, size=self.dim)
                    else:
                        self.population[i] = np.random.uniform(0.0, 5.0) + np.random.normal(0.0, 1.0, size=self.dim)

        return self.population

# One-line description with the main idea
# Adaptive Black Box Optimization using Evolution Strategies