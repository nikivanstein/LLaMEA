import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.iterations = 0
        self.best_individual = None
        self.best_fitness = np.inf
        self.permutation = np.random.permutation(self.dim)

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the point
            value = func(point)
            # Check if the point is within the bounds
            if -5.0 <= point[0] <= 5.0 and -5.0 <= point[1] <= 5.0:
                # If the point is within bounds, update the function value
                self.func_evals += 1
                return value
        # If the budget is exceeded, return the best point found so far
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

    def iterated_permutation(self):
        # Refine the strategy by iterated permutation
        self.iterations += 1
        self.best_individual = np.random.choice([-5.0, 5.0], self.dim, replace=False)
        self.permutation = np.random.permutation(self.dim)
        for _ in range(self.iterations):
            # Select the best individual in the current permutation
            selected_individual = np.random.choice([-5.0, 5.0], self.dim, replace=False)
            # Select the next best individual in the permutation
            next_best_individual = np.random.choice([-5.0, 5.0], self.dim, replace=False)
            # Replace the best individual with the next best individual
            self.best_individual = np.maximum(self.best_individual, selected_individual)
            self.best_individual = np.minimum(self.best_individual, next_best_individual)
            # Swap the best individual with the selected individual
            self.best_individual, selected_individual = selected_individual, self.best_individual
            # Swap the best individual with the next best individual
            self.best_individual, next_best_individual = next_best_individual, self.best_individual
        # Update the function value using the best individual
        value = func(self.best_individual)
        # Check if the best individual is within the bounds
        if -5.0 <= self.best_individual[0] <= 5.0 and -5.0 <= self.best_individual[1] <= 5.0:
            # If the best individual is within bounds, update the function value
            self.func_evals += 1
            return value
        # If the budget is exceeded, return the best point found so far
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 