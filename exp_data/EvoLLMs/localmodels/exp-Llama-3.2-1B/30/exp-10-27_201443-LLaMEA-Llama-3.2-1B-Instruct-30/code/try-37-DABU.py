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
        return f"DABU: {self.budget} evaluations, {self.dim} dimensions"

    def refine_strategy(self, func, search_space):
        # Refine the search space by perturbing the current search space and evaluating the function
        perturbed_search_space = search_space + np.random.normal(0, 1, self.dim)
        func_value = func(perturbed_search_space)
        if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
            return perturbed_search_space
        else:
            # Refine the search space by taking the average of the current search space and the perturbed search space
            new_search_space = (self.search_space + perturbed_search_space) / 2
            return new_search_space

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

def refine_search_space(func, search_space):
    return DABU(1000, 2)  # 1000 function evaluations, 2 dimensions

dabu = DABU(1000, 2)  # 1000 function evaluations, 2 dimensions
print(dabu(test_function))  # prints a random value between -10 and 10

new_search_space = dabu.refine_strategy(test_function, dabu.search_space)
print(new_search_space)