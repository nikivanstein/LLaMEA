import numpy as np
from scipy.optimize import differential_evolution

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None

    def __call__(self, func):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

    def adaptive_black_box(self, func, bounds, initial_point):
        # Define the bounds for the optimization
        lower_bound = bounds[0]
        upper_bound = bounds[1]

        # Define the initial point
        initial_point = np.array(initial_point)

        # Perform the optimization using differential evolution
        result = differential_evolution(func, [(lower_bound, upper_bound), (initial_point - bounds[0], initial_point - bounds[1])], x0=initial_point)

        # Refine the strategy based on the results
        if result.fun < -2 * bounds[0]:
            # If the optimization converges to a lower bound, refine the initial point
            initial_point = np.array(result.x)
        elif result.fun > 2 * bounds[1]:
            # If the optimization converges to an upper bound, refine the initial point
            initial_point = np.array(result.x)

        # Update the function values and the population
        self.func_values = np.zeros(self.dim)
        for _ in range(self.budget):
            idx = np.argmin(np.abs(self.func_values))
            self.func_values[idx] = func(self.func_values[idx])

# Description: AdaptiveBlackBoxOptimizer: A novel metaheuristic algorithm that uses adaptive search and bounds refinement to optimize black box functions.
# Code: 