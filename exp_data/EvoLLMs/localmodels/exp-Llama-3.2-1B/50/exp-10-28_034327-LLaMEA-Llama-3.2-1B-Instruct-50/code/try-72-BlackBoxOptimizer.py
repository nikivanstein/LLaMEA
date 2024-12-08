import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.iterations = 0

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
        self.best_fitness = np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

    def optimize(self, func):
        # Refine the strategy by iteratively changing the individual lines
        for _ in range(self.iterations):
            # Generate a new individual with a modified strategy
            new_individual = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the new individual
            value = func(new_individual)
            # Check if the new individual is better than the best individual found so far
            if value > self.best_fitness:
                # Update the best individual and fitness
                self.best_individual = new_individual
                self.best_fitness = value
                # Update the individual lines to refine the strategy
                self.iterations += 1
                # Change the individual lines to refine the strategy
                new_individual = np.random.uniform(-5.0, 5.0, self.dim)
                new_individual = np.concatenate((new_individual, [self.best_individual[0]] * self.best_fitness / self.best_fitness))
                new_individual = np.concatenate((new_individual, [self.best_individual[1]] * self.best_fitness / self.best_fitness))
                new_individual = np.concatenate((new_individual, [self.best_individual[2]] * self.best_fitness / self.best_fitness))
                new_individual = np.concatenate((new_individual, [self.best_individual[3]] * self.best_fitness / self.best_fitness))
                new_individual = np.concatenate((new_individual, [self.best_individual[4]] * self.best_fitness / self.best_fitness))
                new_individual = np.concatenate((new_individual, [self.best_individual[5]] * self.best_fitness / self.best_fitness))
                new_individual = np.concatenate((new_individual, [self.best_individual[6]] * self.best_fitness / self.best_fitness))
                new_individual = np.concatenate((new_individual, [self.best_individual[7]] * self.best_fitness / self.best_fitness))
                new_individual = np.concatenate((new_individual, [self.best_individual[8]] * self.best_fitness / self.best_fitness))
                new_individual = np.concatenate((new_individual, [self.best_individual[9]] * self.best_fitness / self.best_fitness))
                new_individual = np.concatenate((new_individual, [self.best_individual[10]] * self.best_fitness / self.best_fitness))
                new_individual = np.concatenate((new_individual, [self.best_individual[11]] * self.best_fitness / self.best_fitness))
                new_individual = np.concatenate((new_individual, [self.best_individual[12]] * self.best_fitness / self.best_fitness))
                new_individual = np.concatenate((new_individual, [self.best_individual[13]] * self.best_fitness / self.best_fitness))
                new_individual = np.concatenate((new_individual, [self.best_individual[14]] * self.best_fitness / self.best_fitness))
                new_individual = np.concatenate((new_individual, [self.best_individual[15]] * self.best_fitness / self.best_fitness))
                new_individual = np.concatenate((new_individual, [self.best_individual[16]] * self.best_fitness / self.best_fitness))
                new_individual = np.concatenate((new_individual, [self.best_individual[17]] * self.best_fitness / self.best_fitness))
                new_individual = np.concatenate((new_individual, [self.best_individual[18]] * self.best_fitness / self.best_fitness))
                new_individual = np.concatenate((new_individual, [self.best_individual[19]] * self.best_fitness / self.best_fitness))
                new_individual = np.concatenate((new_individual, [self.best_individual[20]] * self.best_fitness / self.best_fitness))
                new_individual = np.concatenate((new_individual, [self.best_individual[21]] * self.best_fitness / self.best_fitness))
                new_individual = np.concatenate((new_individual, [self.best_individual[22]] * self.best_fitness / self.best_fitness))
                new_individual = np.concatenate((new_individual, [self.best_individual[23]] * self.best_fitness / self.best_fitness))
                new_individual = np.concatenate((new_individual, [self.best_individual[24]] * self.best_fitness / self.best_fitness))
                # Update the individual lines to refine the strategy
                new_individual = np.concatenate((new_individual, [self.iterations * 0.1, self.iterations * 0.1 + 1, self.iterations * 0.1 + 2, self.iterations * 0.1 + 3, self.iterations * 0.1 + 4, self.iterations * 0.1 + 5, self.iterations * 0.1 + 6, self.iterations * 0.1 + 7, self.iterations * 0.1 + 8, self.iterations * 0.1 + 9, self.iterations * 0.1 + 10, self.iterations * 0.1 + 11, self.iterations * 0.1 + 12, self.iterations * 0.1 + 13, self.iterations * 0.1 + 14, self.iterations * 0.1 + 15, self.iterations * 0.1 + 16, self.iterations * 0.1 + 17, self.iterations * 0.1 + 18, self.iterations * 0.1 + 19, self.iterations * 0.1 + 20, self.iterations * 0.1 + 21, self.iterations * 0.1 + 22, self.iterations * 0.1 + 23, self.iterations * 0.1 + 24])
                # Update the individual lines to refine the strategy
                self.func_evals = 0
                # Update the best individual and fitness
                self.best_individual = new_individual
                self.best_fitness = value
        # Return the best individual found so far
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 