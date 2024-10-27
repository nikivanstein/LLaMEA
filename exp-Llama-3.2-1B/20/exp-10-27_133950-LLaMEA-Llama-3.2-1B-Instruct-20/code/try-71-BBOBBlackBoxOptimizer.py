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

    def refine_strategy(self, individual, fitness):
        # Probability 0.2: change individual lines to refine strategy
        if np.random.rand() < 0.2:
            # Change individual lines to refine strategy
            individual[0] = np.random.uniform(-5.0, 5.0, dim)
            individual[1] = np.random.uniform(-5.0, 5.0, dim)
            individual[2] = np.random.uniform(-5.0, 5.0, dim)
        return individual

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)

# Initialize the optimizer with a random individual
individual = [np.random.uniform(-5.0, 5.0, 3) for _ in range(10)]
optimizer = BBOBBlackBoxOptimizer(1000, 10)
result = optimizer(func, individual)
print(result)