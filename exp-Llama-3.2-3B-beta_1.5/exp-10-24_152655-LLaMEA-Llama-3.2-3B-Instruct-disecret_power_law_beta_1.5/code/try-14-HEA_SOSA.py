import numpy as np
import random
import operator

class HEA_SOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0]) * np.ones((dim, 1))
        self.f_best = np.inf
        self.x_best = np.zeros((dim, 1))
        self.f_evals = 0
        self.f_evals_best = 0
        self.population_size = 100
        self.crossover_prob = 0.8
        self.mutation_prob = 0.1
        self.scheduling_prob = 0.018518518518518517

    def __call__(self, func):
        if self.f_evals >= self.budget:
            return self.x_best

        for _ in range(self.budget - self.f_evals):
            # Initialize a list of random candidates
            candidates = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.population_size, self.dim, 1))

            # Evaluate the candidates
            f_candidates = func(candidates)

            # Update the best solution
            f_evals = f_candidates[0]
            x_best = candidates[np.argmin(f_candidates)]

            # Apply Genetic Drift
            x_best = self.drift(x_best)

            # Apply Self-Organizing Scheduling (SOSA)
            x_best = self.sosa(x_best)

            # Update the bounds
            self.bounds = np.array([np.min(candidates, axis=0), np.max(candidates, axis=0)])

            # Update the best solution if necessary
            if f_evals < self.f_best:
                self.f_best = f_evals
                self.x_best = x_best
                self.f_evals_best = f_evals

            # Select the best candidate
            best_candidate = candidates[np.argmin(f_candidates)]

            # Schedule the best candidate
            candidates = np.delete(candidates, np.where(candidates == best_candidate), axis=0)

        return self.x_best

    def drift(self, x):
        # Apply Genetic Drift by perturbing the candidate
        x = x + np.random.normal(0, 0.1, size=x.shape)
        x = np.clip(x, self.bounds[:, 0], self.bounds[:, 1])
        return x

    def sos(self, x):
        # Apply Self-Organizing Scheduling (SOSA) by scheduling the best candidate
        best_candidate = x[np.argmin(np.sum(x**2, axis=1))]
        x = np.delete(x, np.where(x == best_candidate), axis=0)
        return best_candidate

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

hea_sosa = HEA_SOSA(budget=10, dim=2)
x_opt = hea_sosa(func)
print(x_opt)