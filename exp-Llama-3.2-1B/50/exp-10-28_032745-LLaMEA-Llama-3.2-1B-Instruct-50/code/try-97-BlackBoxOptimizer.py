import numpy as np
from scipy.optimize import minimize
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func = lambda x: x[0] * x[1]  # Example black box function

    def __call__(self, func, initial_guess, iterations):
        # Initialize the current best solution
        best_x = initial_guess
        best_value = self.func(best_x)

        # Run the optimization algorithm
        for _ in range(iterations):
            # Evaluate the fitness of the current solution
            fitness = self.func(best_x)

            # Generate new solutions
            new_x = [x + random.uniform(-0.01, 0.01) for x in best_x]
            new_value = self.func(new_x)

            # Check if the new solution is better
            if new_value < best_value:
                # Update the best solution
                best_x = new_x
                best_value = new_value

            # Check if we have reached the budget
            if _ >= self.budget:
                break

        # Return the best solution
        return best_x, best_value

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of the individual
        return self.func(individual)

# Example usage
optimizer = BlackBoxOptimizer(100, 10)
best_individual, best_fitness = optimizer(BlackBoxOptimizer.func, [-5.0, -5.0], 100)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)