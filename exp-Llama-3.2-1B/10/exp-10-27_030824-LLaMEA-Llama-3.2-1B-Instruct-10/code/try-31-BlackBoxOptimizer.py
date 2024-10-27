import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(self.search_space[0], self.search_space[1])
            # Evaluate the function at the point
            evaluation = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Return the point and its evaluation
            return point, evaluation
        else:
            # If the budget is reached, return a default point and evaluation
            return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

    def __str__(self):
        return f"Novel Metaheuristic Algorithm for Black Box Optimization"

    def _random_walk(self, dim, start, end):
        # Perform a random walk in the search space
        return np.random.uniform(start, end, size=(dim,))

    def _linear_interpolation(self, point, target):
        # Perform linear interpolation between the point and the target
        return point + (target - point) * np.linspace(0, 1, 10)

    def _select_next_point(self, budget, points, target):
        # Select the next point based on the budget and the target
        if budget > 0:
            # Choose the point with the highest fitness value
            return np.argmax(points)
        else:
            # If the budget is reached, return a default point and target
            return np.random.uniform(self.search_space[0], self.search_space[1]), target

    def _evaluate_next_point(self, point, target):
        # Evaluate the function at the next point
        return self.func_evaluations + self.func(point)

    def optimize(self, func, budget, dim):
        # Initialize the population with random points
        population = [self._random_walk(dim, -5.0, 5.0) for _ in range(100)]

        # Iterate until the budget is reached
        while self.func_evaluations < budget:
            # Select the next point based on the budget and the target
            next_point = self._select_next_point(budget, population, np.mean(population))

            # Evaluate the function at the next point
            fitness = self._evaluate_next_point(next_point, np.mean(population))

            # Update the population
            population = [self._linear_interpolation(point, fitness) for point in population]

        # Return the best point and its fitness
        return np.mean(population), fitness


# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# Code: 