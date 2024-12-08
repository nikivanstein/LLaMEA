import numpy as np
from scipy.optimize import minimize

class AdaptiveBBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0
        self.step_size = 1.0
        self.iterations = 0

    def __call__(self, func):
        def wrapper(x):
            if self.func_evaluations >= self.budget:
                raise Exception("Optimization limit reached")
            return func(x)

        self.func_evaluations += 1
        try:
            result = minimize(wrapper, self.search_space[0], method="SLSQP", bounds=[(x, x) for x in self.search_space])
            return result.x
        except Exception as e:
            print(f"Error: {e}")
            return None

        self.iterations += 1
        if self.iterations % 10 == 0:
            new_individual = self.refine_individual(wrapper, self.search_space[0], self.iterations)
            self.search_space[0] = new_individual
            self.step_size *= 0.9
            self.step_size = max(0.1, self.step_size)

    def refine_individual(self, func, x, iterations):
        def wrapper(x):
            if x < -5.0 or x > 5.0:
                return func(x)
            return x

        result = minimize(wrapper, x, method="Newton-CG", bounds=[(-5.0, 5.0) for _ in range(self.dim)])
        return result.x

# Example usage:
optimizer = AdaptiveBBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)

# The selected solution to update is AdaptiveBBOBBlackBoxOptimizer