import numpy as np
from scipy.optimize import minimize
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func = lambda x: x[0] * x[1]  # Example black box function

    def __call__(self, func, initial_guess, iterations):
        for _ in range(iterations):
            if _ >= self.budget:
                break
            best_x = initial_guess
            best_value = self.func(best_x)
            for i in range(self.dim):
                new_x = [x + random.uniform(-0.01, 0.01) for x in best_x]
                new_value = self.func(new_x)
                if new_value < best_value:
                    best_x = new_x
                    best_value = new_value
            initial_guess = best_x
        return best_x, best_value

    def novel_search_strategy(self, initial_guess, iterations):
        best_x = initial_guess
        best_value = self.func(best_x)
        for _ in range(iterations):
            new_x = [x + random.uniform(-0.01, 0.01) for x in best_x]
            new_value = self.func(new_x)
            if new_value < best_value:
                best_x = new_x
                best_value = new_value
            if new_value - best_value > 0.45 * best_value:
                # Refine the strategy by changing the step size
                step_size = 0.01
                for i in range(self.dim):
                    new_x[i] += random.uniform(-step_size, step_size)
                best_x = new_x
                best_value = self.func(best_x)
        return best_x, best_value

# Example usage
optimizer = BlackBoxOptimizer(100, 10)
initial_guess = np.array([0, 0])
iterations = 100
best_x, best_value = optimizer(__call__, initial_guess, iterations)
print("Best individual:", best_x)
print("Best value:", best_value)