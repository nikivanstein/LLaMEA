import random
import numpy as np
from scipy.optimize import differential_evolution

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
        return func(x0)

def f(x):
    return x**2 + 0.5*x + 0.1

def f_prime(x):
    return 2*x + 0.5

def f_double_prime(x):
    return 2

def f_double_prime_prime(x):
    return 4

def bbo_opt_metaheuristic(func, x0, bounds, budget):
    # Define the bounds and the bounds' constraints
    bounds = [bounds]
    constraints = []

    # Define the optimization problem using differential evolution
    def objective(x):
        return func(x)

    def constraint(x):
        return [x[i] - bounds[i][0] <= bounds[i][1] for i in range(self.dim)]

    # Perform the optimization using differential evolution
    res = differential_evolution(objective, constraints, x0=x0, bounds=bounds, constraints=constraint, maxiter=self.budget)

    return res.x

# Initialize the metaheuristic algorithm
bbo_opt_metaheuristic_metaheuristic = BBOBMetaheuristic(budget=100, dim=10)

# Evaluate the function using the metaheuristic algorithm
func = lambda x: bbo_opt_metaheuristic_metaheuristic(f, x0=[0.0]*self.dim, bounds=[[-5.0, 5.0]*self.dim], budget=100)
x0 = [0.0]*self.dim
bounds = [[-5.0, 5.0]*self.dim]
res = bbo_opt_metaheuristic_metaheuristic(func, x0=x0, bounds=bounds, budget=100)

# Print the results
print(f"Optimal solution: {res}")
print(f"Optimal fitness: {func(res)}")