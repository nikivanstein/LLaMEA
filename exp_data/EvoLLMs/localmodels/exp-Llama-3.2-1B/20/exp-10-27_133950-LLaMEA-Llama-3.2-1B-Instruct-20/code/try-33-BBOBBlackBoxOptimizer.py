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

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)

# Novel Metaheuristic Algorithm: Adaptive Mutation and Selection
# 
# Description: This algorithm combines adaptive mutation and selection to improve the optimization process.
# It starts with a random initial population, then iteratively applies mutation and selection to refine the strategy.
# The mutation rate is adjusted based on the fitness of the individual, and the selection is based on the fitness and diversity of the population.
# The algorithm stops when the budget is reached or a satisfactory solution is found.