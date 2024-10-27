import random
import numpy as np

class MetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.adaptive_line_search = True

    def __call__(self, func, initial_point, max_iterations=1000, cooling_rate=0.95):
        # Initialize the current point and fitness value
        current_point = initial_point
        fitness_value = self.evaluate_fitness(current_point, func)

        # Generate a random line search direction
        direction = random.uniform(-1, 1)
        for _ in range(max_iterations):
            # Evaluate the function at the current point and direction
            new_point = current_point + direction * 0.1
            new_fitness_value = self.evaluate_fitness(new_point, func)

            # Update the current point and fitness value
            if new_fitness_value < fitness_value + 1e-10:  # arbitrary threshold
                current_point = new_point
                fitness_value = new_fitness_value
            else:
                # If the function has been evaluated within the budget, return the current point
                return current_point

        # If the maximum number of iterations is reached, return the current point as the optimal solution
        if self.adaptive_line_search and fitness_value < 1e-10:  # arbitrary threshold
            return current_point
        else:
            # If the function has been evaluated within the budget, return the point
            return current_point

    def evaluate_fitness(self, point, func):
        # Evaluate the function at the point
        value = func(point)

        # Check if the function has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the current point as the optimal solution
            return point
        else:
            # If the function has been evaluated within the budget, return the point
            return point

# One-line description: "Metaheuristic Optimizer: A novel algorithm that combines random search, function evaluation, and adaptive line search to efficiently solve black box optimization problems"