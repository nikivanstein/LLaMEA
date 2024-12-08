import numpy as np
import random
import operator

class SOSA_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0]) * np.ones((dim, 1))
        self.f_best = np.inf
        self.x_best = np.zeros((dim, 1))
        self.f_evals = 0
        self.f_evals_best = 0
        self.w = 0.8  # inertia weight
        self.pbest = np.zeros((dim, 1))
        self.rnd = np.random.RandomState(42)
        self.psoid = np.zeros((dim, 1))

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

            # Update the personal best
            for i in range(self.dim):
                if f_evals < self.pbest[i, 0]:
                    self.pbest[i, 0] = f_evals
                    self.psoid[i, 0] = candidates[i, 0]

            # Update the global best
            if f_evals < self.f_best:
                self.f_best = f_evals
                self.x_best = candidates[np.argmin(f_candidates)]

            # Update the bounds
            self.bounds = np.array([np.min(candidates, axis=0), np.max(candidates, axis=0)])

            # PSO update
            self.psoid = 0.5 * self.psoid + 0.5 * self.psoid + self.w * np.dot(self.psoid - self.x_best, self.psoid - self.x_best)
            self.psoid = self.psoid / np.linalg.norm(self.psoid)

        return self.x_best

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

sosa_pso = SOSA_PSO(budget=10, dim=2)
x_opt = sosa_pso(func)
print(x_opt)