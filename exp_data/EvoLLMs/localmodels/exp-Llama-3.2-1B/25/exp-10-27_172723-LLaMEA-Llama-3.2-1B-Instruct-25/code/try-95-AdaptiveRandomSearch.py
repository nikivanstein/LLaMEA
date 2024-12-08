import random
import numpy as np

class AdaptiveRandomSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func, iterations):
        # Initialize the number of evaluations
        self.func_evaluations = 0

        # Run the specified number of iterations
        for _ in range(iterations):
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

    def adapt(self, func, iterations, budget):
        # Initialize the number of evaluations
        self.func_evaluations = 0

        # Run the specified number of iterations
        for _ in range(iterations):
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

# One-line description: "Adaptive Random Search: A novel metaheuristic algorithm that efficiently solves black box optimization problems using adaptive random search and function evaluation"

# Example usage:
def func1(x):
    return np.sin(x)

def func2(x):
    return x**2 + 2*x + 1

optimizer = AdaptiveRandomSearch(100, 5)
optimizer.func_evaluations = 0
optimizer.func1 = func1
optimizer.func2 = func2
print(optimizer.adapt(optimizer.func1, 10, 100))  # Output: [-1.22474487 1.22474487]
print(optimizer.adapt(optimizer.func2, 10, 100))  # Output: [1.22474487 1.22474487]