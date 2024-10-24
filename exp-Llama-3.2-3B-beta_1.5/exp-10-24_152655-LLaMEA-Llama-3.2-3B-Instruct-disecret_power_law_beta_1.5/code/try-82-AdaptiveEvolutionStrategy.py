import numpy as np
import random

class AdaptiveEvolutionStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0]) * np.ones((dim, 1))
        self.f_best = np.inf
        self.x_best = np.zeros((dim, 1))
        self.f_evals = 0
        self.f_evals_best = 0
        self.probability = 0.018518518518518517
        self.candidates = None

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
            self.candidates = np.delete(candidates, np.where(candidates == best_candidate), axis=0)

            # Update the bounds
            self.bounds = np.array([np.min(self.candidates, axis=0), np.max(self.candidates, axis=0)])

            # Adaptive probability update
            if self.f_evals / self.budget < self.probability:
                self.probability += 0.001
            elif self.f_evals / self.budget > 1 - self.probability:
                self.probability -= 0.001

            # Randomly select candidates with probability equal to the updated probability
            if random.random() < self.probability:
                candidates = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.dim, 1))

        return self.x_best

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

strategy = AdaptiveEvolutionStrategy(budget=10, dim=2)
x_opt = strategy(func)
print(x_opt)