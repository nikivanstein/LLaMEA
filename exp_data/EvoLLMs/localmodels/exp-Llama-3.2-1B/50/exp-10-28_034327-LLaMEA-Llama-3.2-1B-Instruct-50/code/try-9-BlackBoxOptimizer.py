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
            # Refine the strategy by changing the individual lines of the selected solution
            # to refine its strategy
            if self.iterations % 10 == 0:
                # Change the lower bound to 0
                point[0] = 0
                # Change the upper bound to 10
                point[1] = 10
                # Change the direction of the search space
                if point[0] > 0:
                    point[0] -= 0.1
                else:
                    point[0] += 0.1
                # Change the step size
                self.step_size = 0.1
            self.iterations += 1

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 
# ```python
# import numpy as np
# import random

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
            # Refine the strategy by changing the individual lines of the selected solution
            # to refine its strategy
            # Change the lower bound to 0
            point[0] = 0
            # Change the upper bound to 10
            point[1] = 10
            # Change the direction of the search space
            if point[0] > 0:
                point[0] -= 0.1
            else:
                point[0] += 0.1
            # Change the step size
            self.step_size = 0.1