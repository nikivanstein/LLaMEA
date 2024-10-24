import numpy as np

class GradientDrivenEvolutionarySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = np.random.uniform(-5.0, 5.0, size=dim)
        self.f_best = np.inf
        self.x_best = None
        self.probability = 0.05

    def __call__(self, func):
        for _ in range(self.budget):
            # Compute gradient of the objective function
            gradient = np.zeros(self.dim)
            h = 1e-1
            for i in range(self.dim):
                gradient[i] = (func(self.x + h * np.eye(self.dim)[i]) - func(self.x - h * np.eye(self.dim)[i])) / (2 * h)

            # Update the current solution using evolutionary strategy
            if np.random.rand() < self.probability:
                self.x += 0.5 * np.random.normal(0, 0.1, size=self.dim)
            else:
                self.x += 0.1 * gradient

            # Update the best solution
            f = func(self.x)
            if f < self.f_best:
                self.f_best = f
                self.x_best = self.x.copy()

# Example usage:
def func(x):
    return np.sum(x**2)

gdes = GradientDrivenEvolutionarySearch(budget=1000, dim=10)
gdes("func")