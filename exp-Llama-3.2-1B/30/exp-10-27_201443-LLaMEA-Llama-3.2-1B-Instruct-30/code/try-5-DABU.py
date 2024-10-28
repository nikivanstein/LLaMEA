import numpy as np

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

class BlackBoxOptimizer:
    def __init__(self, budget, dim, alpha=0.3, beta=0.7):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.alpha = alpha
        self.beta = beta

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
            if np.random.rand() < self.alpha:
                # Refine the strategy with probability alpha
                self.search_space = np.linspace(func_value - 1e-6, func_value + 1e-6, self.dim)
            elif np.random.rand() < self.beta:
                # Explore the search space with probability beta
                self.search_space = self.search_space * 1.2
        return func_value

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

optimizer = BlackBoxOptimizer(1000, 2)  # 1000 function evaluations, 2 dimensions
print(optimizer(test_function))  # prints a random value between -10 and 10