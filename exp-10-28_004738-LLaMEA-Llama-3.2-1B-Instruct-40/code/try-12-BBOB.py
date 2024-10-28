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
            func = lambda x: np.random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def __call__(self, func, x0, bounds, budget):
        # Refine the solution by changing individual lines of the strategy
        x = x0
        for _ in range(budget):
            # Evaluate the fitness of the current solution
            fitness = func(x)
            
            # Change individual lines of the strategy to improve efficiency
            if random.random() < 0.4:
                # Change the lower bound
                x = np.clip(x, bounds[0], bounds[1])
            if random.random() < 0.2:
                # Change the upper bound
                x = np.clip(x, bounds[0], bounds[1])
            if random.random() < 0.4:
                # Change the step size
                x = x * np.random.uniform(0.1, 1.0)
            # Evaluate the fitness of the new solution
            fitness = func(x)
            
            # Refine the solution based on the fitness
            if fitness < bounds[0]:
                x = bounds[0]
            elif fitness > bounds[1]:
                x = bounds[1]
            if random.random() < 0.2:
                x = random.uniform(bounds[0], bounds[1])
            if random.random() < 0.4:
                x = random.uniform(bounds[0], bounds[1])
            if random.random() < 0.6:
                # Use the new line of the strategy
                x = func(x)
            # Update the fitness
            fitness = func(x)
            
            # Refine the solution based on the fitness
            if fitness < bounds[0]:
                x = bounds[0]
            elif fitness > bounds[1]:
                x = bounds[1]
            if random.random() < 0.2:
                x = random.uniform(bounds[0], bounds[1])
            if random.random() < 0.4:
                x = random.uniform(bounds[0], bounds[1])
            if random.random() < 0.6:
                # Use the new line of the strategy
                x = func(x)
            # Update the fitness
            fitness = func(x)
        
        return x, fitness

def f(x):
    return x**2 + 0.5*x + 0.1

def f_prime(x):
    return 2*x + 0.5

def f_double_prime(x):
    return 2

def f_double_prime_prime(x):
    return 4

def bbo_opt(func, x0, bounds, budget):
    return BBOB(budget, len(bounds)).__call__(func, x0, bounds, budget)

# Initialize the algorithm
algorithm = bbo_opt(f, [-4.521232642195706], [[-5.0, 5.0], [-5.0, 5.0]], 100)

# Print the initial solution and fitness
print("Initial solution:", algorithm[0])
print("Fitness:", algorithm[1])