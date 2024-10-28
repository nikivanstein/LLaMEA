import random
import numpy as np
from scipy.optimize import minimize

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.funcs = self.generate_functions()

    def generate_functions(self):
        functions = []
        for _ in range(24):
            func = lambda x: random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def __call__(self, func, x0, bounds, budget):
        # Select a random initial solution
        x0 = x0 + np.random.uniform(-0.1, 0.1, self.dim)

        # Perform the specified number of iterations
        for _ in range(budget):
            # Evaluate the fitness of the current solution
            fitness = func(x0, self.funcs)

            # Refine the solution using a probability of 0.4
            x0 = x0 + 0.4 * np.random.uniform(-0.1, 0.1, self.dim)

            # If the solution has improved, stop
            if np.random.rand() < 0.2:
                break

        return x0

def f(x):
    return x**2 + 0.5*x + 0.1

def f_prime(x):
    return 2*x + 0.5

def f_double_prime(x):
    return 2

def f_double_prime_prime(x):
    return 4

def bbo_opt(func, x0, bounds, budget):
    return BBOBMetaheuristic(budget, len(bounds)).__call__(func, x0, bounds, budget)

# Example usage:
problem = RealSingleObjectiveProblem(5, f, f_prime, f_double_prime, f_double_prime_prime)
bounds = [-5.0, 5.0]
x0 = np.array([0.0])
opt = bbo_opt(problem, x0, bounds, 100)

# Print the final solution
print("Optimal solution:", opt)
print("Fitness:", problem.f(opt))