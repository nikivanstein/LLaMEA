import numpy as np
import random

class AdaptiveSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0]) * np.ones((dim, 1))
        self.f_best = np.inf
        self.x_best = np.zeros((dim, 1))
        self.f_evals = 0
        self.f_evals_best = 0
        self.selection_prob = 0.18518518518518517
        self.exploration_prob = 1 - self.selection_prob

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
            selection_mask = np.random.rand(self.dim) < self.selection_prob
            best_candidate = candidates[selection_mask]
            selection_candidates = candidates[~selection_mask]

            # Schedule the best candidate
            if len(best_candidate) > 0:
                candidates = np.delete(candidates, np.where(candidates == best_candidate), axis=0)

            # Explore the remaining candidates
            if len(selection_candidates) > 0:
                exploration_candidates = np.random.choice(selection_candidates, size=int(self.exploration_prob * len(selection_candidates)), replace=False)
                candidates = np.vstack((candidates, exploration_candidates))

            # Update the bounds
            self.bounds = np.array([np.min(candidates, axis=0), np.max(candidates, axis=0)])

        return self.x_best

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

adaptive_sosa = AdaptiveSOSA(budget=10, dim=2)
x_opt = adaptive_sosa(func)
print(x_opt)