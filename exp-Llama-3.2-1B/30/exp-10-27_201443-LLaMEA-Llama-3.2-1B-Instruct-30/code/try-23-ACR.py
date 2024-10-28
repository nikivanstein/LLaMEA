import numpy as np

class ACR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.convergence_rate = 0.5  # initial convergence rate

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
            self.convergence_rate = np.clip(self.convergence_rate * 0.9, 0.01, 0.1)  # adapt convergence rate
        return func_value

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

acr = ACR(1000, 2)  # 1000 function evaluations, 2 dimensions
print(acr(test_function))  # prints a random value between -10 and 10