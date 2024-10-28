import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
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
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

class IteratedPermutationCooling(BlackBoxOptimizer):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        # Initialize the population with random points
        population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(100)]

        # Initialize the current best point and its fitness
        current_best = population[0]
        current_best_fitness = self.evaluate_fitness(current_best)

        # Iterate for the specified number of iterations
        for _ in range(self.iterations):
            # Generate a list of points within the bounds
            points = [point for point in population if -5.0 <= point[0] <= 5.0 and -5.0 <= point[1] <= 5.0]

            # Select the fittest points to replace the current best point
            fittest_points = random.sample(points, len(points) // 2)
            fittest_points.sort(key=lambda point: self.evaluate_fitness(point), reverse=True)

            # Replace the current best point with the fittest point
            new_individual = fittest_points[0]
            new_individual_fitness = self.evaluate_fitness(new_individual)

            # Update the current best point and its fitness
            current_best = new_individual
            current_best_fitness = new_individual_fitness

            # Add the new individual to the population
            population.append(new_individual)

        # Return the best point found after the specified number of iterations
        return current_best

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 