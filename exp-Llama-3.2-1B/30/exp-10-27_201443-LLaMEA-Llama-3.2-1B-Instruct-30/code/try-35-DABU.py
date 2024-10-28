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

    def adapt_search_space(self, func_values):
        # Refine the search space based on the convergence rate of the current algorithm
        # This line is modified to refine the strategy
        avg_func_value = np.mean(func_values)
        if avg_func_value < 1.0:  # convergence rate is low
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        elif avg_func_value > 1.0:  # convergence rate is high
            self.search_space = np.linspace(1.0, 5.0, self.dim)
        else:
            # Use a weighted average of the current and previous search spaces
            weights = [0.5, 0.3, 0.2]  # 50% of the time use the previous search space, 30% use the current search space, 20% use a random search space
            self.search_space = np.random.choice(self.search_space, p=weights)

    def __str__(self):
        return f"Budget: {self.budget}, Dim: {self.dim}, Search Space: {self.search_space}"

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

dabu = DABU(1000, 2)  # 1000 function evaluations, 2 dimensions
print(dabu(test_function))  # prints a random value between -10 and 10

# Select a solution to update
selected_solution = dabu

# Update the algorithm with the selected solution
new_dabu = DABU(1000, 2)  # 1000 function evaluations, 2 dimensions
new_dabu.budget = selected_solution.budget  # keep the same budget
new_dabu.dim = selected_solution.dim  # keep the same dimensionality
new_dabu.search_space = selected_solution.search_space  # keep the same search space
new_dabu.func_evaluations = 0  # reset the number of function evaluations
new_dabu.adapt_search_space([func(test_function(x)) for x in np.random.rand(1000, 2)])  # adapt the search space

# Print the updated algorithm
print(new_dabu)