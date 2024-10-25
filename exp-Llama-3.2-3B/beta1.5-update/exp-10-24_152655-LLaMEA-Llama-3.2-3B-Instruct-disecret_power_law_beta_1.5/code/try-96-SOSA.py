import numpy as np
import random

class SOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0]) * np.ones((dim, 1))
        self.f_best = np.inf
        self.x_best = np.zeros((dim, 1))
        self.f_evals = 0
        self.f_evals_best = 0
        self.selection_prob = 0.4074074074074074

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

            # Select the best candidate using probability-based selection
            indices = np.argsort(f_candidates)
            selection_indices = np.random.choice(indices, size=int(self.selection_prob * len(indices)), replace=False)
            selected_candidates = candidates[selection_indices]

            # Update the best solution if necessary
            if f_evals < self.f_best:
                self.f_best = f_evals
                self.x_best = selected_candidates[0]
                self.f_evals_best = f_evals

            # Update the bounds
            self.bounds = np.array([np.min(selected_candidates, axis=0), np.max(selected_candidates, axis=0)])

            # Perform mutation using probability-based selection
            mutation_indices = np.random.choice(len(selected_candidates), size=int(self.selection_prob * len(selected_candidates)), replace=False)
            mutated_candidates = selected_candidates.copy()
            mutated_candidates[mutation_indices] = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(len(mutation_indices), 1))
            selected_mutated_candidates = mutated_candidates[np.argsort(f_candidates[mutation_indices])]

            # Update the bounds
            self.bounds = np.array([np.min(selected_mutated_candidates, axis=0), np.max(selected_mutated_candidates, axis=0)])

        return self.x_best