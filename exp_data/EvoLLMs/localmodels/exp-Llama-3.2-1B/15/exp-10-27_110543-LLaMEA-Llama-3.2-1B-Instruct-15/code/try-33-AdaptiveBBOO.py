import numpy as np
from scipy.optimize import minimize

class AdaptiveBBOO:
    def __init__(self, budget, dim, refinement_ratio=0.15):
        self.budget = budget
        self.dim = dim
        self.refinement_ratio = refinement_ratio
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = []
        self.algorithms = []

    def __call__(self, func):
        def eval_func(x):
            return func(x)

        def evaluate_budget(func, x, budget):
            if budget <= 0:
                raise ValueError("Budget cannot be zero or negative")
            return np.sum([eval_func(x + np.random.normal(0, 1, size=self.dim)) for _ in range(budget)])

        def _optimize(func, budget, dim):
            return np.optimize.minimize(lambda x: evaluate_budget(func, x, budget), x0=np.random.uniform(-5.0, 5.0, size=(dim,)), method='SLSQP', bounds=[(-5.0, 5.0)] * (dim,))

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
            next_population = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                x = self.population[i]
                if np.random.rand() < self.refinement_ratio:
                    x = _optimize(eval_func, self.budget, self.dim)(x)
                next_population[i] = x + np.random.normal(0, 1, size=self.dim)
                fitness = evaluate_budget(eval_func, next_population[i], self.budget)
                next_population[i] = next_population[i][np.argsort(self.fitnesses, axis=0)]
                self.population[i] = next_population[i]

        # Update the selected solution
        selected_solution = self.population[np.argsort(self.fitnesses, axis=0)]
        selected_solution = selected_solution[np.argsort(self.fitnesses, axis=0)]

        self.algorithms.append((selected_solution, eval_func, evaluate_budget))

        return selected_solution

# One-line description with the main idea
# Adaptive Black Box Optimization using Evolution Strategies with Refinement
# 