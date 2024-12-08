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
        self.probability = 0.037037037037037035

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
            probabilities = np.array([np.exp(-np.abs(f_candidate - f_evals_best)) for f_candidate in f_candidates])
            selected_index = np.random.choice(len(f_candidates), p=probabilities)
            selected_candidate = candidates[selected_index]

            # Schedule the selected candidate
            candidates = np.delete(candidates, selected_index, axis=0)

            # Update the bounds
            if selected_candidate < self.bounds[:, 0].min():
                self.bounds[:, 0] = np.maximum(self.bounds[:, 0], selected_candidate)
            elif selected_candidate > self.bounds[:, 1].max():
                self.bounds[:, 1] = np.minimum(self.bounds[:, 1], selected_candidate)

        return self.x_best

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

sosa = SOSA(budget=10, dim=2)
x_opt = sosa(func)
print(x_opt)