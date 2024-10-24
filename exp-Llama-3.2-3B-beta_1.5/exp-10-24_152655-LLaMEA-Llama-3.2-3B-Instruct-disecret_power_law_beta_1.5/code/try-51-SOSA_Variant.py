import numpy as np
import random

class SOSA_Variant:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0]) * np.ones((dim, 1))
        self.f_best = np.inf
        self.x_best = np.zeros((dim, 1))
        self.f_evals = 0
        self.f_evals_best = 0
        self.selection_prob = 0.07407407407407407

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
            selection_mask = np.random.rand(len(candidates)) < self.selection_prob
            best_candidate = candidates[selection_mask]

            # Schedule the best candidate
            candidates = np.delete(candidates, np.where(candidates == best_candidate), axis=0)

            # Update the bounds
            self.bounds = np.array([np.min(candidates, axis=0), np.max(candidates, axis=0)])

            # Perform mutation with a probability of 0.1
            if np.random.rand() < 0.1:
                # Randomly select two candidates
                candidate1 = np.random.choice(candidates, 1, replace=False)
                candidate2 = np.random.choice(candidates, 1, replace=False)

                # Perform crossover
                x_crossover = (candidate1 + candidate2) / 2

                # Update the bounds
                self.bounds = np.array([np.min([candidate1[0], candidate2[0], x_crossover[0]]), np.max([candidate1[0], candidate2[0], x_crossover[0]])])

                # Replace the candidates with the crossover result
                candidates = np.delete(candidates, np.where(candidates == candidate1), axis=0)
                candidates = np.delete(candidates, np.where(candidates == candidate2), axis=0)
                candidates = np.append(candidates, x_crossover)

        return self.x_best

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

sosa_variant = SOSA_Variant(budget=10, dim=2)
x_opt = sosa_variant(func)
print(x_opt)