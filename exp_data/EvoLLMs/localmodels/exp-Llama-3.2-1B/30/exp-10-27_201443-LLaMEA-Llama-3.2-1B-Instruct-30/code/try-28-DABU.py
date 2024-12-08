import numpy as np
from scipy.optimize import minimize_scalar

class DABU:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.iterations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return func_value

    def adapt_sampling(self):
        if self.iterations < 100:
            self.iterations += 1
            if np.random.rand() < 0.3:
                self.search_space = np.linspace(0.0, 5.0, self.dim)
            else:
                self.search_space = np.linspace(-5.0, 0.0, self.dim)
        else:
            self.search_space = np.linspace(0.0, 5.0, self.dim)

    def refine(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return func_value

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

dabu = DABU(1000, 2)  # 1000 function evaluations, 2 dimensions
print(dabu(test_function))  # prints a random value between -10 and 10

# Initial solution
initial_solution = np.random.uniform(-5.0, 5.0, (2,))

# Run the DABU algorithm
best_func_value = np.inf
best_func = None
for _ in range(100):
    func_value = dabu(test_function)
    if func_value < best_func_value:
        best_func_value = func_value
        best_func = dabu(refine(test_function))(initial_solution)
    dabu.adapt_sampling()

# Print the final solution
print(best_func)