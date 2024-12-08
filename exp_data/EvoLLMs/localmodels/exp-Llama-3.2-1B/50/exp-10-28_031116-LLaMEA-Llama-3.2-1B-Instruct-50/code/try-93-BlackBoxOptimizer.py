import random
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.generate_population()

    def generate_population(self):
        return [(np.random.uniform(-5.0, 5.0), np.random.uniform(-5.0, 5.0)) for _ in range(self.population_size)]

    def __call__(self, func):
        def search_func(x, func):
            return func(x)

        # Randomly select an initial point from the population
        x0 = random.choice(self.population)

        # Perform a specified number of function evaluations within the budget
        num_evals = min(self.budget, len(self.population))
        results = [search_func(x0, func) for _ in range(num_evals)]

        # Refine the solution based on the results
        refined_x0 = self.budget_results(x0, results, num_evals)

        # Evaluate the refined solution using the original function
        refined_func = self.evaluate_func(refined_x0, func)

        # Return the refined solution and its score
        return refined_func, refined_x0

    def budget_results(self, x0, results, num_evals):
        # Refine the solution by iteratively applying a probabilistic search strategy
        x = x0
        for _ in range(num_evals):
            # Choose a point with probability 0.45
            if random.random() < 0.45:
                x = x0
            # Choose a point with probability 0.55
            else:
                # Apply a perturbation to the current point
                perturbation = np.random.uniform(-1.0, 1.0)
                x += perturbation
            # Evaluate the function at the new point
            func_value = self.evaluate_func(x, func)
            # Update the solution with the highest function value
            x0 = x
        return x

    def evaluate_func(self, x, func):
        return func(x)