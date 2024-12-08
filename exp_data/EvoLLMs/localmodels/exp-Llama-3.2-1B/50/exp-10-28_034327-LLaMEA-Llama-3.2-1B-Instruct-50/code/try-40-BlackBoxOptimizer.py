import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.new_individuals = []

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
        # Select the best individual in the population
        best_individual = self.new_individuals[0]
        # Select the worst individual in the population
        worst_individual = self.new_individuals[-1]
        # Refine the strategy by changing the worst individual
        for _ in range(10):
            # Select two random individuals
            individual1 = np.random.choice(self.new_individuals, 1)
            individual2 = np.random.choice(self.new_individuals, 1)
            # Refine the worst individual
            worst_individual = np.argmin([func(individual1), func(individual2)])
        # Update the new individuals
        self.new_individuals = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.dim)]
        return best_individual, worst_individual

    def cooling(self):
        # Apply the cooling schedule
        self.new_individuals = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.dim)]
        return self.new_individuals

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 