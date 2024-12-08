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
        if fitness < 0.2:
            # Change individual lines to refine strategy
            individual[0] = np.random.uniform(0, 1)
            individual[1] = np.random.uniform(0, 1)
            individual[2] = np.random.uniform(0, 1)
        else:
            # Keep individual lines as is
            pass

        # Update fitness
        fitness *= 1.1

        # Check for optimization limit
        if fitness > 1.0:
            raise Exception("Optimization limit exceeded")

        return individual, fitness

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)

# Refine the solution
refined_individual, refined_fitness = optimizer.refine_strategy(result, func)
print(refined_individual)
print(refined_fitness)