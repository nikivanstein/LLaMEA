import numpy as np
import random

class EvolutionaryGradient:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = np.random.uniform(-5.0, 5.0, size=dim)
        self.f_best = np.inf
        self.x_best = None
        self.refinement_probability = 0.025

    def __call__(self, func):
        for _ in range(self.budget):
            # Compute gradient of the objective function
            gradient = np.zeros(self.dim)
            h = 1e-1
            for i in range(self.dim):
                gradient[i] = (func(self.x + h * np.eye(self.dim)[i]) - func(self.x - h * np.eye(self.dim)[i])) / (2 * h)

            # Update the current solution using evolutionary strategy
            self.x += 0.5 * np.random.normal(0, 0.1, size=self.dim)

            # Update the best solution
            f = func(self.x)
            if f < self.f_best:
                self.f_best = f
                self.x_best = self.x.copy()

            # Add gradient information to the evolutionary strategy
            self.x += 0.1 * gradient

            # Refine the solution with probabilistic refinement
            if random.random() < self.refinement_probability:
                # Randomly select two individuals to refine
                indices = np.random.choice(self.dim, size=2, replace=False)
                new_x = self.x.copy()
                for i in indices:
                    # Perturb the selected dimension
                    new_x[i] += np.random.uniform(-0.1, 0.1)
                # Evaluate the new solution
                new_f = func(new_x)
                # Update the best solution if the new solution is better
                if new_f < self.f_best:
                    self.f_best = new_f
                    self.x_best = new_x.copy()

# Example usage:
def func(x):
    return np.sum(x**2)

evg = EvolutionaryGradient(budget=1000, dim=10)
evg("func")
