import random
import numpy as np

class BBOB:
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

    def __call__(self, func):
        return func(random.uniform(-5.0, 5.0))

def f(x):
    return x**2 + 0.5*x + 0.1

def f_prime(x):
    return 2*x + 0.5

def f_double_prime(x):
    return 2

def f_double_prime_prime(x):
    return 4

def bbo_opt(func, x0, bounds, budget):
    x = x0
    for _ in range(budget):
        x = func(x)
        if x < bounds[0]:
            x = bounds[0]
        elif x > bounds[1]:
            x = bounds[1]
        if random.random() < 0.4:  # 40% chance to change line search direction
            a = np.random.uniform(-1, 1)
            b = np.random.uniform(-1, 1)
            c = np.random.uniform(-1, 1)
            d = np.random.uniform(-1, 1)
            new_x = x + a * (x - x) + b * (x_prime - x) + c * (x_prime - x_prime) + d * (x_prime_prime - x_prime)
            if random.random() < 0.2:  # 20% chance to change direction of gradient descent
                new_x = x - a * (f_prime(x) - f(x_prime)) - b * (f_prime_prime(x) - f_prime(x_prime)) + c * (f_prime_prime_prime(x) - f_prime_prime(x_prime))
        if random.random() < 0.2:  # 20% chance to change direction of line search
            new_x = x + a * (f(x) - f(bounds[0])) + b * (f(x_prime) - f(bounds[0])) + c * (f(x_prime_prime) - f(x_prime)) + d * (f(x_prime_prime_prime) - f(x_prime_prime))
        if random.random() < 0.4:  # 40% chance to refine the solution
            new_x = x + a * (f(x) - f(bounds[0])) + b * (f(x_prime) - f(bounds[0])) + c * (f(x_prime_prime) - f(x_prime)) + d * (f(x_prime_prime_prime) - f(x_prime_prime))
        x = new_x
        if random.random() < 0.2:  # 20% chance to stop early
            break
    return x

# Example usage:
algorithm = BBOB(budget=100, dim=5)
problem = RealSingleObjectiveProblem(1.0, 5.0, Sphere, iid=True)
best_individual = algorithm(problem, -4.521232642195706)
print("Best individual:", best_individual)
print("Best fitness:", algorithm(problem, best_individual))