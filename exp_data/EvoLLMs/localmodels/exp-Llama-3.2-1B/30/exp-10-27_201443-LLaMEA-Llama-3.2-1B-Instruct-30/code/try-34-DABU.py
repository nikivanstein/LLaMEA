import numpy as np

class DABU:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.search_space_history = self.search_space.copy()

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
            self.search_space_history.append(self.search_space.copy())

        # Refine the search space based on the function evaluations
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.search_space = np.array([self.search_space])
        self.search_space = self.search_space[np.argsort(self.search_space_evaluations)]

        # Add the most recent search space to the history
        self.search_space_history.append(self.search_space.copy())

        return func_value

    def search_space_evaluations(self):
        return np.array(self.search_space_evaluations)

# One-line description with the main idea
# DABU: Dynamic Adaptive Black Box Optimization using Dynamic Search Space Refining

# Description: DABU uses a dynamic search space to adaptively refine its search space based on the function evaluations, allowing for efficient exploration-exploitation tradeoff.
# Code:
# ```python
# DABU: Dynamic Adaptive Black Box Optimization using Dynamic Search Space Refining
# ```
# ```python
# def test_function(x):
#     return np.exp(-x[0]**2 - x[1]**2)

# dabu = DABU(1000, 2)  # 1000 function evaluations, 2 dimensions
# print(dabu(test_function))  # prints a random value between -10 and 10