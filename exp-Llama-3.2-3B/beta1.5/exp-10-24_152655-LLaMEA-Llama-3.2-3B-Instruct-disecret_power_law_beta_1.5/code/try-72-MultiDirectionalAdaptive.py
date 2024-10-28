import numpy as np
import random

class MultiDirectionalAdaptive:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0]) * np.ones((dim, 1))
        self.f_best = np.inf
        self.x_best = np.zeros((dim, 1))
        self.f_evals = 0
        self.f_evals_best = 0
        self.directions = np.random.uniform(-1, 1, size=(self.dim, 1))

    def __call__(self, func):
        if self.f_evals >= self.budget:
            return self.x_best

        for _ in range(self.budget - self.f_evals):
            # Evaluate the current best solution
            f_evals = func(self.x_best)

            # Generate new directions
            new_directions = np.random.uniform(-1, 1, size=(self.dim, 1))

            # Compute new candidate solutions
            candidates = self.x_best + self.directions * np.random.uniform(0, 1, size=(self.dim, 1))

            # Evaluate the new candidate solutions
            f_candidates = func(candidates)

            # Update the best solution if necessary
            if f_evals > f_candidates[0]:
                self.f_best = f_evals
                self.x_best = candidates[np.argmin(f_candidates)]
                self.f_evals_best = f_evals

            # Update the directions
            self.directions = self.directions * 0.9 + new_directions * 0.1

            # Update the bounds
            self.bounds = np.array([np.min(candidates, axis=0), np.max(candidates, axis=0)])

        return self.x_best

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

mda = MultiDirectionalAdaptive(budget=10, dim=2)
x_opt = mda(func)
print(x_opt)