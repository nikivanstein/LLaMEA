import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0

    def __call__(self, func):
        # Initialize the population with random points in the search space
        population = [(random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1])) for _ in range(100)]

        # Evaluate the function at each point in the population
        for _ in range(self.budget):
            # Generate a new point by mutating the current point
            mutated_point = (self.search_space[0] + random.uniform(-self.search_space[1], self.search_space[1]) / 2, self.search_space[0] + random.uniform(-self.search_space[1], self.search_space[1]) / 2)
            # Evaluate the function at the mutated point
            func_value = func(mutated_point)
            # Check if the mutated point is within the budget
            if self.func_evaluations < self.budget:
                # If not, replace the worst point in the population with the mutated point
                population[self.func_evaluations] = mutated_point
                # Update the best point found so far
                best_point = max(population, key=lambda x: x[1])
                if func_value < best_point[1]:
                    population[self.func_evaluations] = best_point
                self.func_evaluations += 1
            else:
                # If the budget is reached, return the best point found so far
                return best_point

        # Return the best point found
        return self.search_space[0], self.search_space[1]

# Example usage:
optimizer = BlackBoxOptimizer(100, 10)
best_solution = optimizer(100)
print(best_solution)