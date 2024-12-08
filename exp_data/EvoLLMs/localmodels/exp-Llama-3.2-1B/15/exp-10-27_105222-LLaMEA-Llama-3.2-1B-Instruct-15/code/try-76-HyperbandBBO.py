import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class HyperbandBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim
        self.bayes = False

    def __call__(self, func):
        if self.bayes:
            # Sample a new point in the search space using Bayesian optimization
            x = np.random.uniform(*self.search_space, size=self.search_space_dim)
            # Evaluate the function at the new point using Bayesian optimization
            func_value = func(x)
            # Store the function value and the new point
            self.func_evals += 1
            self.func_evals_evals = func_value
            # Store the new point in the search space
            self.search_space = (min(self.search_space[0], x), max(self.search_space[1], x))
        else:
            # Sample a new point in the search space using Gaussian distribution
            x = np.random.uniform(*self.search_space, size=self.search_space_dim)
            # Evaluate the function at the new point
            func_value = func(x)
            # Store the function value and the new point
            self.func_evals += 1
            self.func_evals_evals = func_value
            # Store the new point in the search space
            self.search_space = (min(self.search_space[0], x), max(self.search_space[1], x))

    def bayes_optimize(self, func, bounds, num_points):
        # Perform Bayesian optimization
        result = minimize(lambda x: -func(x), x0=np.random.uniform(*bounds), method="SLSQP", bounds=bounds, options={'maxiter': 1000})
        return result.x, -result.fun

    def __call__(self, func):
        while self.func_evals < self.budget:
            if self.bayes:
                # Perform Bayesian optimization
                x, _ = self.bayes_optimize(func, self.search_space, self.dim)
            else:
                # Sample a new point in the search space using Gaussian distribution
                x = np.random.uniform(*self.search_space, size=self.search_space_dim)
                # Evaluate the function at the new point
                func_value = func(x)
                # Store the function value and the new point
                self.func_evals += 1
                self.func_evals_evals = func_value
                # Store the new point in the search space
                self.search_space = (min(self.search_space[0], x), max(self.search_space[1], x))
        # Evaluate the function at the final point in the search space
        func_value = func(self.search_space)
        return func_value