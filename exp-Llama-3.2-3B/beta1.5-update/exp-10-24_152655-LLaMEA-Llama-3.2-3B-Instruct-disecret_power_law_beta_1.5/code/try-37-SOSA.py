import numpy as np
import random
import operator

class SOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0]) * np.ones((dim, 1))
        self.f_best = np.inf
        self.x_best = np.zeros((dim, 1))
        self.f_evals = 0
        self.f_evals_best = 0
        self.p = 0.3888888888888889

    def __call__(self, func):
        if self.f_evals >= self.budget:
            return self.x_best

        for _ in range(self.budget - self.f_evals):
            # Initialize a list of random candidates
            candidates = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.dim, 1))

            # Evaluate the candidates
            f_candidates = func(candidates)

            # Calculate the probability of selecting each candidate
            probabilities = np.exp(-(f_candidates - self.f_best) / (2 * self.p))
            probabilities /= np.sum(probabilities)

            # Select the best candidate based on the calculated probabilities
            best_candidate_idx = np.random.choice(len(candidates), p=probabilities)
            best_candidate = candidates[best_candidate_idx]

            # Schedule the best candidate
            candidates = np.delete(candidates, np.where(candidates == best_candidate), axis=0)

            # Update the bounds
            self.bounds = np.array([np.min(candidates, axis=0), np.max(candidates, axis=0)])

            # Update the best solution if necessary
            f_evals = f_candidates[best_candidate_idx]
            if f_evals < self.f_best:
                self.f_best = f_evals
                self.x_best = best_candidate
                self.f_evals_best = f_evals

        return self.x_best

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

sosa = SOSA(budget=10, dim=2)
x_opt = sosa(func)
print(x_opt)