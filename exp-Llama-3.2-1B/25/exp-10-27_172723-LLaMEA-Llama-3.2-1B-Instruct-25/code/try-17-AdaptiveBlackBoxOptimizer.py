import random
import numpy as np
import math

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.best_individual = None
        self.best_fitness = float('inf')

    def __call__(self, func):
        # Evaluate the function for the specified number of times
        num_evaluations = min(self.budget, self.func_evaluations + 1)
        func_evaluations = self.func_evaluations
        self.func_evaluations += num_evaluations

        # Generate a random point in the search space
        point = np.random.choice(self.search_space)

        # Evaluate the function at the point
        value = func(point)

        # Check if the function has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the current point as the optimal solution
            return point
        else:
            # If the function has been evaluated within the budget, return the point
            return point

    def adapt_search(self):
        # If the best solution has not been found yet, start with a random point
        if self.best_individual is None:
            self.best_individual = self.evaluate_fitness(np.array([0.0]*self.dim))
            self.best_fitness = self.evaluate_fitness(self.best_individual)

        # Generate a new point based on the best solution
        new_point = self.best_individual + (self.best_individual - self.best_individual) * self.budget / self.func_evaluations

        # Evaluate the new point
        new_value = self.evaluate_fitness(new_point)

        # Check if the new point is better than the current best solution
        if new_value < self.best_fitness:
            # If so, update the best solution and its fitness
            self.best_individual = new_point
            self.best_fitness = new_value

# One-line description: "Adaptive Black Box Optimizer: A novel metaheuristic algorithm that adapts its search strategy based on the performance of previous solutions"

# Initialize the optimizer with a budget and dimension
optimizer = AdaptiveBlackBoxOptimizer(100, 10)

# Run the optimizer for 100 iterations
for _ in range(100):
    # Optimize the black box function
    solution = optimizer(func, 100)

    # Print the solution and its fitness
    print(f"Solution: {solution}, Fitness: {optimizer.evaluate_fitness(solution)}")

    # Update the best solution
    optimizer.adapt_search()