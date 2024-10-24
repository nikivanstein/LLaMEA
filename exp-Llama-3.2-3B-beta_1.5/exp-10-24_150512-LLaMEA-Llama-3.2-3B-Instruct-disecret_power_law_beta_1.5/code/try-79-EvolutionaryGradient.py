import numpy as np
import random
from scipy.optimize import minimize

class EvolutionaryGradient:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = np.random.uniform(-5.0, 5.0, size=dim)
        self.f_best = np.inf
        self.x_best = None

    def __call__(self, func):
        for _ in range(self.budget):
            # Compute gradient of the objective function
            gradient = np.zeros(self.dim)
            h = 1e-1
            for i in range(self.dim):
                gradient[i] = (func(self.x + h * np.eye(self.dim)[i]) - func(self.x - h * np.eye(self.dim)[i])) / (2 * h)

            # Update the current solution using evolutionary strategy
            self.x += 0.5 * np.random.normal(0, 0.1, size=self.dim)

            # Add probability 0.025 to change the individual lines of the evolutionary strategy
            if random.random() < 0.025:
                mutation = np.random.uniform(-0.5, 0.5, size=self.dim)
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

# Alternative approach using scipy's minimize function
# ```python
# import numpy as np
# import random
# from scipy.optimize import minimize

class EvolutionaryGradient:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = np.random.uniform(-5.0, 5.0, size=dim)
        self.f_best = np.inf
        self.x_best = None

    def __call__(self, func):
        for _ in range(self.budget):
            # Define the objective function to minimize
            def objective(x):
                return func(x)

            # Define the gradient of the objective function
            def gradient(x):
                h = 1e-1
                return np.array([ (func(x + h * np.eye(self.dim)[i]) - func(x - h * np.eye(self.dim)[i])) / (2 * h) for i in range(self.dim) ])

            # Minimize the objective function using scipy's minimize function
            res = minimize(objective, self.x, method="SLSQP", jac=gradient)

            # Update the current solution
            self.x = res.x

            # Update the best solution
            f = func(self.x)
            if f < self.f_best:
                self.f_best = f
                self.x_best = self.x.copy()

# Example usage:
def func(x):
    return np.sum(x**2)

evg = EvolutionaryGradient(budget=1000, dim=10)
evg("func")