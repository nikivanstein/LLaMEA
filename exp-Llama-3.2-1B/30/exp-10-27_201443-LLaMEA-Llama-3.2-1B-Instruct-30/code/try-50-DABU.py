import numpy as np
import random

class DABU:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return func_value

    def __str__(self):
        return f"Budget: {self.budget}, Dim: {self.dim}, Search Space: {self.search_space}"

class DABUWithProbabilityOfChange:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.change_probability = 0.3

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        if random.random() < self.change_probability:
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
            self.func_evaluations = 0
        return func_value

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

dabu_with_probability = DABUWithProbabilityOfChange(1000, 2)  # 1000 function evaluations, 2 dimensions
print(dabu_with_probability(test_function))  # prints a random value between -10 and 10

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 