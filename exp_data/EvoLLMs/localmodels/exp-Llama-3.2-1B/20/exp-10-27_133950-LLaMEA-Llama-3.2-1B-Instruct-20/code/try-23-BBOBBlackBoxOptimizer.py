import numpy as np
from scipy.optimize import minimize

class BBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0

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

    def adapt(self, new_individual):
        # Refine the strategy by changing the individual lines of the selected solution
        # to refine its strategy
        # (0.2 probability)
        if np.random.rand() < 0.2:
            new_individual.lines = [np.random.uniform(-5.0, 5.0, dim) for dim in self.dim]
        else:
            new_individual.lines = [np.random.uniform(-5.0, 5.0, dim) for dim in self.dim]

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)

# Update the optimizer with the new solution
new_individual = result
new_individual.adapt()
print(new_individual)