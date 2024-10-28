import numpy as np
import random

class EvolutionaryGradientEntropy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = np.random.uniform(-5.0, 5.0, size=dim)
        self.f_best = np.inf
        self.x_best = None
        self.entropy = 0.0

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

            # Update entropy-based exploration
            self.entropy = np.mean(np.abs(self.x - self.x_best))
            if self.entropy > 0.125:
                # Randomly select a fraction of the population to explore
                indices = np.random.choice(self.dim, size=int(0.125 * self.dim), replace=False)
                self.x += 0.05 * np.random.choice([-1, 1], size=len(indices), p=[0.5, 0.5])

            # Check for convergence
            if _ % 100 == 0 and np.all(np.abs(self.x - self.x_best) < 1e-6):
                print("Converged after {} iterations".format(_))

# Example usage:
def func(x):
    return np.sum(x**2)

evg = EvolutionaryGradientEntropy(budget=1000, dim=10)
evg("func")