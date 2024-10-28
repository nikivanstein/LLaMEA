import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

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

class IteratedPermutationCooling:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population = None

    def __call__(self, func):
        # Initialize the population with random points in the search space
        self.population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(100)]

        # Iterate over the population until the budget is exceeded
        while self.func_evals < self.budget:
            # Initialize the best point and its value
            best_point = None
            best_value = np.inf

            # Iterate over the population
            for individual in self.population:
                # Evaluate the function at the individual
                value = func(individual)

                # Check if the individual is within the bounds
                if -5.0 <= individual[0] <= 5.0 and -5.0 <= individual[1] <= 5.0:
                    # If the individual is within bounds, update its value
                    value = np.max(func(individual))

                # Update the best point and its value if necessary
                if value < best_value:
                    best_point = individual
                    best_value = value

            # Update the population with the best points
            self.population = [best_point if best_point is not None else np.random.uniform(-5.0, 5.0, self.dim) for _ in range(100)]

            # Update the population size
            self.population = self.population[:100]

            # Update the function evaluations
            self.func_evals += 1

        # Return the best point found
        return np.max(func(self.population[0]))

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 