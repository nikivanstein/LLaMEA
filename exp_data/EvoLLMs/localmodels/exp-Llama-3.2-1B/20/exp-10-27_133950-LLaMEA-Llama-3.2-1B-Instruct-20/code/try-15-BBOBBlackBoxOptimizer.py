import numpy as np
from scipy.optimize import minimize, differential_evolution
from collections import deque

class BBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0
        self.population_size = 100
        self.population_deque = deque(maxlen=self.population_size)
        self.population_history = []

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

        # Select next individual based on probability
        probabilities = np.random.rand(self.population_size, self.dim)
        next_individual_index = np.argmax(probabilities)
        next_individual = self.population_deque[next_individual_index]
        self.population_deque.append(next_individual)

        # Refine strategy by changing individual lines
        if np.random.rand() < 0.2:
            next_individual.lines = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.dim)]

        return next_individual

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)

# Update population history
optimizer.population_history.append((result, func))