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

    def update_strategy(self):
        if self.func_evaluations < self.budget // 3:
            # Refine strategy by increasing the upper bound
            self.search_space = np.linspace(5.0, 10.0, self.dim)
            self.func_evaluations = 0
        elif self.func_evaluations < self.budget // 2:
            # Refine strategy by decreasing the lower bound
            self.search_space = np.linspace(0.0, -5.0, self.dim)
            self.func_evaluations = 0
        else:
            # Refine strategy by increasing the upper bound
            self.search_space = np.linspace(5.0, 10.0, self.dim)
            self.func_evaluations = 0

    def __str__(self):
        return f"DABU with {self.budget} function evaluations, {self.dim} dimensions"

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

dabu = DABU(1000, 2)  # 1000 function evaluations, 2 dimensions
print(dabu(test_function))  # prints a random value between -10 and 10

dabu.update_strategy()
print(dabu)  # prints DABU with 500 function evaluations, 2 dimensions