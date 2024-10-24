import numpy as np
import random

class PSOA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0]) * np.ones((dim, 1))
        self.f_best = np.inf
        self.x_best = np.zeros((dim, 1))
        self.f_evals = 0
        self.f_evals_best = 0
        self.swarm = []
        self.candidates = []
        self.particle_swarm = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.dim, 1))

    def __call__(self, func):
        if self.f_evals >= self.budget:
            return self.x_best

        for _ in range(self.budget - self.f_evals):
            # Initialize a list of random candidates
            self.candidates = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.dim, 1))

            # Evaluate the candidates
            f_candidates = func(self.candidates)

            # Update the best solution
            f_evals = f_candidates[0]
            x_best = self.candidates[0]
            f_evals_best = f_evals

            # Update the best solution if necessary
            if f_evals < self.f_best:
                self.f_best = f_evals
                self.x_best = x_best
                self.f_evals_best = f_evals

            # Select the best candidate
            best_candidate = self.candidates[np.argmin(f_candidates)]

            # Schedule the best candidate
            self.swarm.append(best_candidate)
            self.swarm = np.delete(self.swarm, np.where(self.swarm == best_candidate), axis=0)

            # Update the bounds
            self.bounds = np.array([np.min(self.candidates, axis=0), np.max(self.candidates, axis=0)])

            # Update the particle swarm
            for i in range(len(self.swarm)):
                r1 = random.random()
                r2 = random.random()
                if r1 < 0.018518518518518517:
                    self.particle_swarm[i] += self.swarm[i] - self.particle_swarm[i]
                if r2 < 0.018518518518518517:
                    self.particle_swarm[i] += random.uniform(-1, 1) * (self.swarm[i] - self.particle_swarm[i])

            # Update the bounds
            self.bounds = np.array([np.min(self.particle_swarm, axis=0), np.max(self.particle_swarm, axis=0)])

        return self.x_best

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

psoa = PSOA(budget=10, dim=2)
x_opt = psoa(func)
print(x_opt)