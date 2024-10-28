import numpy as np
import random

class DABU:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.tuning_parameters = [0.3, 0.7]  # probability of changing individual lines

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return func_value

    def tune(self, func, initial_tuning_parameters):
        for i in range(len(initial_tuning_parameters)):
            # Randomly select an individual line to change
            line_to_change = np.random.choice(len(self.search_space), 1)
            new_tuning_parameters = initial_tuning_parameters.copy()
            new_tuning_parameters[i] += random.uniform(-0.1, 0.1)
            # Calculate the new objective value
            new_func_value = func(self.search_space)
            # Update the objective function evaluations
            self.func_evaluations += 1
            # If the new objective value is better, update the objective function evaluations
            if np.abs(new_func_value) < 1e-6:
                self.func_evaluations += 1

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

dabu = DABU(1000, 2)  # 1000 function evaluations, 2 dimensions
initial_tuning_parameters = [0.3, 0.7]

print("Original DABU:", dabu(test_function))

# Tune the DABU algorithm
dabu.tune(test_function, initial_tuning_parameters)

print("Tuned DABU:", dabu(test_function))