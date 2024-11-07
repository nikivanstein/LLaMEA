import numpy as np
from scipy.optimize import differential_evolution
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

class LeverageExploration:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.x_best = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
        self.f_best = np.inf
        self.bayesian_space = {
            'bounds': [(self.lower_bound, self.upper_bound) for _ in range(self.dim)],
            'type': ['uniform'] * self.dim
        }

    def __call__(self, func):
        for _ in range(self.budget):
            # Explore
            x_explore = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
            f_explore = func(x_explore)

            # Local search
            x_local = self.x_best
            f_local = func(x_local)

            # Leverage the best exploration point
            if f_explore < f_local:
                self.x_best = x_explore
                self.f_best = f_explore

            # Differential evolution for local search
            x_dev = differential_evolution(func, [(self.lower_bound, self.upper_bound) for _ in range(self.dim)])
            f_dev = func(x_dev.x)

            # Update the best point if the local search is better
            if f_dev < self.f_best:
                self.x_best = x_dev.x
                self.f_best = f_dev

            # Bayesian optimization for adaptive exploration
            @use_named_args(self.bayesian_space)
            def objective(params):
                return func(params)

            result = gp_minimize(objective, self.x_best, n_calls=10, verbose=False)
            x_bayes = result.x

            # Update the best point if the Bayesian optimization is better
            if np.min([func(x) for x in [x_explore, x_local, x_dev.x, x_bayes]]) < self.f_best:
                self.x_best = x_bayes
                self.f_best = np.min([func(x) for x in [x_explore, x_local, x_dev.x, x_bayes]])

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

le = LeverageExploration(budget=100, dim=2)
le(func)
print("Best point:", le.x_best)
print("Best function value:", le.f_best)