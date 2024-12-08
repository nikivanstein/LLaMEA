import numpy as np
import random

class SOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0]) * np.ones((dim, 1))
        self.elite_size = int(self.budget * 0.1)
        self.elite = np.zeros((self.elite_size, dim))
        self.f_best = np.inf
        self.x_best = np.zeros((dim, 1))
        self.f_evals = 0
        self.f_evals_best = 0
        self.elite_size_history = []

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
            x_best = candidates[np.argmin(f_candidates)]

            # Update the bounds
            self.bounds = np.array([np.min(candidates, axis=0), np.max(candidates, axis=0)])
            self.elite_size_history.append(len(self.elite))

            # Update the elite
            self.elite = np.delete(self.elite, np.where(self.elite == x_best), axis=0)
            self.elite = np.vstack((self.elite, x_best))

            # Select the best candidate
            best_candidate = x_best

            # Schedule the best candidate
            candidates = np.delete(candidates, np.where(candidates == best_candidate), axis=0)

            # Update the bounds
            self.bounds = np.array([np.min(candidates, axis=0), np.max(candidates, axis=0)])

            # Update the best solution if necessary
            if f_evals < self.f_best:
                self.f_best = f_evals
                self.x_best = best_candidate
                self.f_evals_best = f_evals

        # Update the elite size
        if len(self.elite) > self.elite_size:
            self.elite = self.elite[:self.elite_size]

        # Update the elite
        self.elite = np.vstack((self.elite, self.x_best))

        return self.x_best

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

sosa = SOSA(budget=10, dim=2)
x_opt = sosa(func)
print(x_opt)