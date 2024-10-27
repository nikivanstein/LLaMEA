import numpy as np
from scipy.optimize import minimize

class AdaptiveUpperBoundSampling(BBOBBlackBoxOptimizer):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0

    def __call__(self, func):
        def wrapper(x):
            if self.func_evaluations >= self.budget:
                raise Exception("Optimization limit reached")
            if np.random.rand() < 0.2:
                # Refine strategy by changing the individual lines
                updated_individual = self.evaluate_fitness(self.search_space[0], func)
            else:
                # Use the previous individual
                updated_individual = self.evaluate_fitness(self.search_space[0], func)
            self.func_evaluations += 1
            try:
                result = minimize(wrapper, updated_individual, method="SLSQP", bounds=[(x, x) for x in self.search_space])
                return result.x
            except Exception as e:
                print(f"Error: {e}")
                return None

# Example usage:
optimizer = AdaptiveUpperBoundSampling(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)