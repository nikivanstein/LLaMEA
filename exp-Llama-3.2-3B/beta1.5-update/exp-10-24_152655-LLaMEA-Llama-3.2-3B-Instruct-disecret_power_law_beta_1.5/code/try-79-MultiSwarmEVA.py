import numpy as np
import random
import operator

class MultiSwarmEVA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0]) * np.ones((dim, 1))
        self.f_best = np.inf
        self.x_best = np.zeros((dim, 1))
        self.f_evals = 0
        self.f_evals_best = 0
        self.swarms = [np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.dim, 1)) for _ in range(5)]

    def __call__(self, func):
        if self.f_evals >= self.budget:
            return self.x_best

        for _ in range(self.budget - self.f_evals):
            # Evaluate the swarms
            f_swarm = np.array([func(s) for s in self.swarms])

            # Update the best solution
            f_evals = f_swarm[0]
            x_best = self.swarms[0]
            f_evals_best = f_evals

            # Update the best solution if necessary
            if f_evals < self.f_best:
                self.f_best = f_evals
                self.x_best = x_best
                self.f_evals_best = f_evals

            # Update the swarms
            for i in range(len(self.swarms)):
                # Select the best candidate
                best_candidate = self.swarms[i][np.argmin(f_swarm[i:i+1])]

                # Schedule the best candidate
                new_swarm = np.delete(self.swarms[i], np.where(self.swarms[i] == best_candidate), axis=0)

                # Add a new candidate
                new_swarm = np.vstack((new_swarm, best_candidate))

                # Update the bounds
                self.bounds = np.array([np.min(new_swarm, axis=0), np.max(new_swarm, axis=0)])

                # Update the swarms
                self.swarms[i] = new_swarm

        return self.x_best

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

ms Eva = MultiSwarmEVA(budget=10, dim=2)
x_opt = ms Eva(func)
print(x_opt)