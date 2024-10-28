import numpy as np
from scipy.optimize import minimize_scalar

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
        return f"DABU: Optimizing {self.dim}-dimensional function using {self.budget} evaluations"

    def __repr__(self):
        return f"DABU(budget={self.budget}, dim={self.dim})"

    def optimize(self, func):
        # Refine the search space based on the results of the first evaluation
        if self.func_evaluations == 0:
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        else:
            # Use the previously optimized search space as a starting point
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
            for _ in range(10):
                # Use a random search within the current search space
                new_search_space = np.random.uniform(self.search_space)
                # Evaluate the function at the new search space
                func_value = func(new_search_space)
                # If the function value is close to zero, stop exploring this branch
                if np.abs(func_value) < 1e-6:
                    break
                # Otherwise, update the search space for the next iteration
                self.search_space = new_search_space

        # Perform a random search within the refined search space
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

# Refine the search space based on the results of the first evaluation
dabu.optimize(test_function)

# Update the algorithm with the refined search space
dabu = DABU(1000, 2)  # 1000 function evaluations, 2 dimensions
print(dabu(test_function))  # prints a random value between -10 and 10

# Evaluate the function using the updated algorithm
dabu.optimize(test_function)