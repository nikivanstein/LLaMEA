import numpy as np
import random

class EvolutionarySwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0]) * np.ones((dim, 1))
        self.f_best = np.inf
        self.x_best = np.zeros((dim, 1))
        self.f_evals = 0
        self.f_evals_best = 0
        self.swarm = self.initialize_swarm()

    def __call__(self, func):
        if self.f_evals >= self.budget:
            return self.x_best

        for _ in range(self.budget - self.f_evals):
            # Evaluate the swarm
            f_swarm = func(self.swarm)

            # Update the best solution
            f_evals = f_swarm[0]
            x_best = self.swarm[0]
            f_evals_best = f_evals

            # Update the best solution if necessary
            if f_evals < self.f_best:
                self.f_best = f_evals
                self.x_best = x_best
                self.f_evals_best = f_evals

            # Update the swarm
            self.swarm = self.update_swarm(f_swarm, self.bounds)

        return self.x_best

    def initialize_swarm(self):
        swarm = np.zeros((self.budget, self.dim, 1))
        for i in range(self.budget):
            swarm[i] = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(1, self.dim, 1))
        return swarm

    def update_swarm(self, f_swarm, bounds):
        # Select the best individual
        best_individual = f_swarm[np.argmin(f_swarm)]

        # Update the swarm
        new_swarm = np.delete(swarm, np.where(swarm == best_individual), axis=0)
        new_swarm = np.concatenate((new_swarm, np.random.uniform(bounds[:, 0], bounds[:, 1], size=(self.budget - len(new_swarm), self.dim, 1))))
        new_swarm = np.sort(new_swarm, axis=0)
        return new_swarm

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

eso = EvolutionarySwarmOptimization(budget=10, dim=2)
x_opt = eso(func)
print(x_opt)