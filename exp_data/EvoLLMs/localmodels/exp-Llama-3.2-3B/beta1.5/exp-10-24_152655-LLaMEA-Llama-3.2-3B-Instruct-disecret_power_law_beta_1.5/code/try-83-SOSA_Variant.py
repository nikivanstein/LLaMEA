import numpy as np
import random
import operator

class SOSA_Variant:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0]) * np.ones((dim, 1))
        self.f_best = np.inf
        self.x_best = np.zeros((dim, 1))
        self.f_evals = 0
        self.f_evals_best = 0
        self.probability = 0.05555555555555555

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

            # Select the best candidate using probability-based selection
            selection_mask = np.random.rand(self.dim) < self.probability
            best_candidate = candidates[selection_mask]
            candidates = np.delete(candidates, np.where(candidates == best_candidate), axis=0)

            # Update the bounds
            self.bounds = np.array([np.min(candidates, axis=0), np.max(candidates, axis=0)])

        return self.x_best

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

sosa_variant = SOSA_Variant(budget=10, dim=2)
x_opt = sosa_variant(func)
print(x_opt)