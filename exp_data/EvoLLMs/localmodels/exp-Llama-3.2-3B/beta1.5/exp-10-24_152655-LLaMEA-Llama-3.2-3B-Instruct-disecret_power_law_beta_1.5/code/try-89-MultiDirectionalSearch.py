import numpy as np
import random

class MultiDirectionalSearch:
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
            # Initialize a list of random candidates
            candidates = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.dim, 1))

            # Evaluate the candidates
            f_candidates = func(candidates)

            # Update the best solution
            f_evals = f_candidates[0]
            x_best = candidates[0]
            f_evals_best = f_evals

            # Update the best solution if necessary
            if f_evals < self.f_best:
                self.f_best = f_evals
                self.x_best = x_best
                self.f_evals_best = f_evals

            # Perform multi-directional search
            for direction in self.directions:
                # Generate new candidates by adding the direction to the current best candidate
                new_candidates = candidates + direction * 1.0

                # Evaluate the new candidates
                f_new_candidates = func(new_candidates)

                # Update the best solution if necessary
                f_evals_new = f_new_candidates[0]
                x_best_new = new_candidates[0]
                f_evals_best_new = f_evals_new

                if f_evals_new < f_evals_best:
                    self.f_best = f_evals_new
                    self.x_best = x_best_new
                    self.f_evals_best = f_evals_best_new

            # Update the bounds
            self.bounds = np.array([np.min(candidates, axis=0), np.max(candidates, axis=0)])

            # Update the directions
            self.directions = np.random.uniform(-1, 1, size=(self.dim, 1))

        return self.x_best

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

mds = MultiDirectionalSearch(budget=10, dim=2)
x_opt = mds(func)
print(x_opt)