import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None

    def __call__(self, func):
        if self.func_values is None:
            # Refine strategy: Use adaptive sampling to balance exploration and exploitation
            # Initialize a random sample of dimensions from the search space
            self.func_values = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the initial sample
            func(self.func_values)
        else:
            while self.func_evals > 0:
                # Use the current sample to guide the next evaluation
                idx = np.argmin(np.abs(self.func_values))
                # Refine the sample by adding a small perturbation to the current point
                self.func_values[idx] += np.random.uniform(-0.1, 0.1)
                # Evaluate the function at the refined sample
                func(self.func_values)
                # Increment the evaluation count
                self.func_evals -= 1
                # If all evaluations are completed, break the loop
                if self.func_evals == 0:
                    break

# Description: AdaptiveBlackBoxOptimizer: A metaheuristic algorithm for solving black box optimization problems using adaptive sampling and perturbation.
# Code: 
# ```python
# import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None

    def __call__(self, func):
        if self.func_values is None:
            # Refine strategy: Use adaptive sampling to balance exploration and exploitation
            # Initialize a random sample of dimensions from the search space
            self.func_values = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the initial sample
            func(self.func_values)
        else:
            while self.func_evals > 0:
                # Use the current sample to guide the next evaluation
                idx = np.argmin(np.abs(self.func_values))
                # Refine the sample by adding a small perturbation to the current point
                self.func_values[idx] += np.random.uniform(-0.1, 0.1)
                # Evaluate the function at the refined sample
                func(self.func_values)
                # Increment the evaluation count
                self.func_evals -= 1
                # If all evaluations are completed, break the loop
                if self.func_evals == 0:
                    break