import random
import numpy as np
from scipy.optimize import differential_evolution

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func, initial_point=None, iterations=100, step_size=0.01):
        if initial_point is None:
            initial_point = np.random.uniform(self.search_space[0], self.search_space[1])

        for _ in range(iterations):
            # Evaluate the function at the current point
            evaluation = func(initial_point)

            # If the budget is reached, return a default point and evaluation
            if self.func_evaluations >= self.budget:
                return np.random.uniform(self.search_space[0], self.search_space[1]), evaluation

            # Perform gradient descent to refine the strategy
            gradient = np.array([func(i) - evaluation for i in initial_point])
            gradient /= np.linalg.norm(gradient)
            step_size = self.step_size * gradient
            initial_point -= step_size

        # If the budget is not reached, return a default point and evaluation
        return np.random.uniform(self.search_space[0], self.search_space[1]), evaluation

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.
# ```python
# BlackBoxOptimizer: Novel Metaheuristic Algorithm for Black Box Optimization
# ```