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
        self.dim_strategies = {
            'uniform': np.random.uniform,
            'gaussian': np.random.normal,
            'bounded': lambda x: np.clip(x, -5.0, 5.0)
        }

    def __call__(self, func):
        def eval_func(x):
            return func(x)

        def evaluate_budget(func, x, budget):
            if budget <= 0:
                raise ValueError("Budget cannot be zero or negative")
            return np.sum([eval_func(x + np.random.normal(0, 1, size=self.dim)) for _ in range(budget)])

        def select_strategy(x):
            strategy = np.random.choice(list(self.dim_strategies.keys()))
            return self.dim_strategies[strategy](x)

        def mutate(x):
            return select_strategy(x) + np.random.normal(0, 1, size=self.dim)

        for _ in range(self.budget):
            fitness = evaluate_budget(eval_func, self.population, self.budget)
            self.fitnesses[np.argsort(self.fitnesses, axis=0)] = fitness
            self.population_history.append(self.population)

        # Select the fittest individuals
        self.population = self.population[np.argsort(self.fitnesses, axis=0)]
        self.fitnesses = self.fitnesses[np.argsort(self.fitnesses, axis=0)]

        # Evolve the population
        for _ in range(100):
            next_population = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                next_population[i] = mutate(self.population[i])
                fitness = evaluate_budget(eval_func, next_population[i], self.budget)
                next_population[i] = next_population[i][np.argsort(self.fitnesses, axis=0)]
                self.population[i] = next_population[i]

        return self.population

# One-line description with the main idea
# Adaptive Black Box Optimization using Evolution Strategies