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

            # Select the best candidate
            best_candidate = candidates[np.argmin(f_candidates)]

            # Schedule the best candidate
            candidates = np.delete(candidates, np.where(candidates == best_candidate), axis=0)

            # Update the bounds
            self.bounds = np.array([np.min(candidates, axis=0), np.max(candidates, axis=0)])

            # Adaptive mutation
            if self.f_evals > self.budget * 0.2:
                mutation_rate = 0.1
            else:
                mutation_rate = self.probability

            # Randomly mutate the candidates
            mutated_candidates = candidates + np.random.uniform(-1, 1, size=(self.dim, 1)) * (1 - mutation_rate)

            # Ensure the mutated candidates are within the bounds
            mutated_candidates = np.clip(mutated_candidates, self.bounds[:, 0], self.bounds[:, 1])

            # Evaluate the mutated candidates
            f_mutated = func(mutated_candidates)

            # Update the best solution if necessary
            if np.min(f_mutated) < f_evals_best:
                self.f_best = np.min(f_mutated)
                self.x_best = mutated_candidates[np.argmin(f_mutated)]

        return self.x_best

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

sosa = SOSA(budget=10, dim=2)
x_opt = sosa(func)
print(x_opt)