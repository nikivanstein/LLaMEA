import numpy as np
import random

class AdaptiveResamplingBasedEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.resampling_rate = 0.1
        self.resampling_counter = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            if self.resampling_counter < self.resampling_rate * self.budget:
                self.resampling_counter += 1
                self.search_space = np.linspace(-5.0, 5.0, self.dim)
            else:
                self.search_space = np.random.choice(self.search_space, size=self.dim, replace=False)
        return func_value

# Initialize the algorithm with a budget of 1000 function evaluations and a dimension of 10
algorithm = AdaptiveResamplingBasedEvolutionaryAlgorithm(budget=1000, dim=10)

# Evaluate the function 1000 times
function_value = algorithm(func)

# Print the result
print("Function value:", function_value)