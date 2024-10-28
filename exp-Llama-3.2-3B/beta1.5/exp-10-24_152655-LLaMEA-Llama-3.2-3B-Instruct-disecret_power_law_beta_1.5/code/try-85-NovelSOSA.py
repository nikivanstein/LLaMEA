import numpy as np
import random

class NovelSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0]) * np.ones((dim, 1))
        self.f_best = np.inf
        self.x_best = np.zeros((dim, 1))
        self.f_evals = 0
        self.f_evals_best = 0
        self.mutation_prob = 0.18518518518518517
        self.candidates = np.zeros((self.budget, self.dim, 1))

    def __call__(self, func):
        if self.f_evals >= self.budget:
            return self.x_best

        for i in range(self.budget):
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

            # Store the candidates for future use
            self.candidates[i] = candidates

            # Select the best candidate
            best_candidate = candidates[np.argmin(f_candidates)]

            # Schedule the best candidate
            candidates = np.delete(candidates, np.where(candidates == best_candidate), axis=0)

            # Update the bounds
            self.bounds = np.array([np.min(candidates, axis=0), np.max(candidates, axis=0)])

            # Apply mutation with adaptive probability
            if random.random() < self.mutation_prob:
                # Randomly select a candidate to mutate
                idx = random.randint(0, self.budget - 1)
                # Mutate the selected candidate
                mutated_idx = np.random.randint(0, self.dim)
                self.candidates[idx, mutated_idx, 0] += np.random.uniform(-1, 1)

        return self.x_best

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

novel_sosa = NovelSOSA(budget=10, dim=2)
x_opt = novel_sosa(func)
print(x_opt)