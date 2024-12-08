import numpy as np
import random

class EvolutionaryGradient:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = np.random.uniform(-5.0, 5.0, size=dim)
        self.f_best = np.inf
        self.x_best = None
        self.probability = 0.025

    def __call__(self, func):
        for _ in range(self.budget):
            # Compute gradient of the objective function
            gradient = np.zeros(self.dim)
            h = 1e-1
            for i in range(self.dim):
                gradient[i] = (func(self.x + h * np.eye(self.dim)[i]) - func(self.x - h * np.eye(self.dim)[i])) / (2 * h)

            # Update the current solution using evolutionary strategy
            mutation = np.random.normal(0, 0.1, size=self.dim)
            if random.random() < self.probability:
                mutation *= 0.5
            self.x += mutation

            # Update the best solution
            f = func(self.x)
            if f < self.f_best:
                self.f_best = f
                self.x_best = self.x.copy()

            # Add gradient information to the evolutionary strategy
            self.x += 0.1 * gradient

            # Check for convergence
            if _ % 100 == 0 and np.all(np.abs(self.x - self.x_best) < 1e-6):
                print("Converged after {} iterations".format(_))

# Example usage:
def func(x):
    return np.sum(x**2)

evg = EvolutionaryGradient(budget=1000, dim=10)
evg("func")