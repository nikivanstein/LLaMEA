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

    def refine_strategy(self, individual):
        # Probability 0.2: Refine the strategy by changing the individual lines of the selected solution
        if random.random() < 0.2:
            # Change the lower bound of the individual
            individual[0] = np.random.uniform(0, 10)
            print(f"Refined lower bound: {individual[0]}")

        # Change the upper bound of the individual
        individual[1] = np.random.uniform(5, 15)
        print(f"Refined upper bound: {individual[1]}")

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)

# Initialize a new individual with a refined strategy
individual = [np.random.uniform(0, 10), np.random.uniform(5, 15)]
optimizer.refine_strategy(individual)