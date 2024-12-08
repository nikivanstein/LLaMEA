import random
import numpy as np
import copy

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.iterations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget and self.iterations < 1000:
            # Refine the strategy
            self.iterations += 1
            # Generate a random point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            func_value = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Check if the point is within the budget
            if self.func_evaluations < self.budget:
                # If not, return the point
                return point
            # Change the strategy
            if random.random() < 0.15:
                # If 15% chance, change the direction of the search
                if random.random() < 0.5:
                    # If 50% chance, change the direction to the left
                    direction = (-1, 0)
                else:
                    # If 50% chance, change the direction to the right
                    direction = (1, 0)
                # Update the search direction
                self.search_space[0] += direction[0] * 0.1
                self.search_space[1] += direction[1] * 0.1
            # Change the point
            point = (point[0] + random.uniform(-0.1, 0.1), point[1] + random.uniform(-0.1, 0.1))
            # Evaluate the function at the new point
            func_value = func(point)
            # Update the best point found so far
            if func_value < self.search_space[0] or func_value > self.search_space[1]:
                self.search_space = (self.search_space[0], self.search_space[1])
            # Update the best point found so far
            self.search_space = (point[0], point[1])
        # If the budget is reached, return the best point found so far
        return self.search_space[0], self.search_space[1]

# Example usage
if __name__ == "__main__":
    optimizer = BlackBoxOptimizer(100, 5)
    func = lambda x: np.sin(x)
    best_point, best_func_value = optimizer(func)
    print(f"Best point: {best_point}, Best function value: {best_func_value}")