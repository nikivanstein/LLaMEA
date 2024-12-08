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
        self.iterations = 0

    def __call__(self, func):
        def eval_func(x):
            return func(x)

        def evaluate_budget(func, x, budget):
            if budget <= 0:
                raise ValueError("Budget cannot be zero or negative")
            return np.sum([eval_func(x + np.random.normal(0, 1, size=self.dim)) for _ in range(budget)])

        def optimize_func(x, budget):
            def fitness(x):
                return evaluate_budget(eval_func, x, budget)
            return minimize(fitness, x, method="SLSQP", bounds=[(-5.0, 5.0)] * self.dim, options={"maxiter": self.budget})

        self.population = optimize_func(self.population, self.budget)
        return self.population

# One-line description with the main idea
# Adaptive Black Box Optimization using Evolution Strategies

# Code: 