import numpy as np
import random

class EvolutionaryGradientAdaptation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = np.random.uniform(-5.0, 5.0, size=dim)
        self.f_best = np.inf
        self.x_best = None
        self.gradient_history = []
        self.adaptation_rate = 0.125

    def __call__(self, func):
        for _ in range(self.budget):
            # Compute gradient of the objective function
            gradient = np.zeros(self.dim)
            h = 1e-1
            for i in range(self.dim):
                gradient[i] = (func(self.x + h * np.eye(self.dim)[i]) - func(self.x - h * np.eye(self.dim)[i])) / (2 * h)

            # Update the current solution using evolutionary strategy
            self.x += 0.5 * np.random.normal(0, 0.1, size=self.dim)

            # Add gradient information to the evolutionary strategy
            self.x += 0.1 * gradient

            # Check for convergence
            if _ % 100 == 0 and np.all(np.abs(self.x - self.x_best) < 1e-6):
                print("Converged after {} iterations".format(_))

            # Adapt the evolutionary strategy
            if random.random() < self.adaptation_rate:
                # Randomly select a dimension to adapt
                dim_to_adapt = random.randint(0, self.dim - 1)
                # Adapt the evolutionary strategy in the selected dimension
                self.x[dim_to_adapt] += np.random.uniform(-0.5, 0.5)

            # Update the best solution
            f = func(self.x)
            if f < self.f_best:
                self.f_best = f
                self.x_best = self.x.copy()

            # Store the current gradient for adaptation
            self.gradient_history.append(gradient)

            # Adapt the evolutionary strategy using the gradient history
            if len(self.gradient_history) > 10:
                avg_gradient = np.mean(self.gradient_history, axis=0)
                self.x += 0.05 * avg_gradient

# Example usage:
def func(x):
    return np.sum(x**2)

evg_adapt = EvolutionaryGradientAdaptation(budget=1000, dim=10)
evg_adapt("func")