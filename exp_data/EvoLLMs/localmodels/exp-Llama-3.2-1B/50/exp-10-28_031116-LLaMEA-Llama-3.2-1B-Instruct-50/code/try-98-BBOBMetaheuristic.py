import numpy as np
import random
from scipy.optimize import minimize

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []

    def __call__(self, func):
        # Evaluate the function within the search space
        func_min, func_max = np.min(func), np.max(func)
        func_evals = np.random.randint(func_min, func_max + 1, self.budget)

        # Initialize the population with random solutions
        for _ in range(self.budget):
            x = np.random.uniform(-5.0, 5.0, self.dim)
            y = np.random.uniform(-5.0, 5.0, self.dim)
            self.population.append((x, y))

        # Select the best solution based on the budget
        selected_solution = self.select_solution(self.population, func_evals)

        # Optimize the selected solution using the given function
        def objective(x):
            return func(x[0], x[1])

        def bounds(x):
            return [x[0] - 5.0, x[0] + 5.0, x[1] - 5.0, x[1] + 5.0]

        res = minimize(objective, selected_solution, method="SLSQP", bounds=bounds, constraints={"type": "eq", "fun": lambda x: x[0] + x[1]}, options={"maxiter": 1000})

        # Update the population with the new solution
        self.population = [x for x, y in self.population if y[0] + y[1] == res.x[0] + res.x[1]]

        # Evaluate the new function evaluations
        func_evals = np.random.randint(func_min, func_max + 1, self.budget)
        self.population = [(x, y) for x, y in self.population if y[0] + y[1] == res.x[0] + res.x[1]]

        return res.fun

    def select_solution(self, population, func_evals):
        # Use the probability 0.45 to refine the strategy
        weights = np.random.rand(len(population))
        selected_solution = population[np.random.choice(len(population), p=weights)]
        return selected_solution