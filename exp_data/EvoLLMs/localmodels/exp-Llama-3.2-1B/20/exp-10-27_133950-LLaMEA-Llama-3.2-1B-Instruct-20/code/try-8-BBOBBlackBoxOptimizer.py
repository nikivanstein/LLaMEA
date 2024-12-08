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

    def select_strategy(self):
        # Novel Metaheuristic Algorithm: "Adaptive Step Size"
        # Refine the strategy by changing the step size based on the fitness value
        if self.func_evaluations == 0:
            return self.search_space

        step_size = np.sqrt(self.func_evaluations / self.budget)
        return self.search_space + [x + i * step_size for i, x in enumerate(self.search_space)]

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)

# Update the solution using the selected strategy
updated_individual = optimizer.select_strategy()
print(updated_individual)