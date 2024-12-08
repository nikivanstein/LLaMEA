import numpy as np
from scipy.optimize import differential_evolution, minimize
import random

class LeverageExploration:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.x_best = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
        self.f_best = np.inf
        self.T = 1000  # initial temperature
        self.alpha = 0.99  # cooling rate
        self.step_size = 0.1  # initial step size
        self.step_size_adaptation = 0.5  # adaptation rate for step size
        self.gamma = 0.8  # new cooling schedule parameter

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

            # Simulated annealing for exploration
            if np.random.rand() < 0.5:
                x_simsa = self.x_best + np.random.uniform(-self.step_size, self.step_size, size=self.dim)
                f_simsa = func(x_simsa)
                if f_simsa < f_explore:
                    self.x_best = x_simsa
                    self.f_best = f_simsa

            # Cooling schedule with adaptive step size
            self.T *= self.alpha
            if self.T < 1:
                self.T = 1
            self.step_size *= (1 - self.step_size_adaptation)
            if self.step_size < 0.01:
                self.step_size = 0.01
            self.T *= self.gamma  # New cooling schedule parameter

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

le = LeverageExploration(budget=100, dim=2)
le(func)
print("Best point:", le.x_best)
print("Best function value:", le.f_best)