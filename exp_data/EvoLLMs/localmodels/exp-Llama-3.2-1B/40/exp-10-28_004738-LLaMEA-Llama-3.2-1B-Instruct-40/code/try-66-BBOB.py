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

    def __call__(self, func, initial_guess, bounds, mutation_rate, adaptive_line_search=True, line_search_step=0):
        x = initial_guess
        for _ in range(self.budget):
            if adaptive_line_search:
                x = bbo_opt(func, x, bounds, self.budget, line_search_step)
            else:
                x = func(x)
            if x < bounds[0]:
                x = bounds[0]
            elif x > bounds[1]:
                x = bounds[1]
            if random.random() < 0.5:
                x = random.uniform(bounds[0], bounds[1])
            if random.random() < 0.2:
                x = random.uniform(bounds[0], bounds[1])
            if random.random() < 0.4:
                x = random.uniform(bounds[0], bounds[1])
        return x

def f(x):
    return x**2 + 0.5*x + 0.1

def f_prime(x):
    return 2*x + 0.5

def f_double_prime(x):
    return 2

def f_double_prime_prime(x):
    return 4

def bbo_opt(func, x0, bounds, budget, mutation_rate=0.01, adaptive_line_search=True, line_search_step=0):
    x = x0
    for _ in range(budget):
        if adaptive_line_search:
            x = bbo_opt(func, x, bounds, budget, line_search_step)
        else:
            x = func(x)
        if x < bounds[0]:
            x = bounds[0]
        elif x > bounds[1]:
            x = bounds[1]
        if random.random() < mutation_rate:
            x = random.uniform(bounds[0], bounds[1])
        if random.random() < 0.2:
            x = random.uniform(bounds[0], bounds[1])
        if random.random() < 0.4:
            x = random.uniform(bounds[0], bounds[1])
    return x

# Initialize the algorithm
bbo = BBOB(100, 10)
bbo.funcs = bbo.generate_functions()

# Evaluate the objective function 100 times
bbo.evaluate_fitness(f, bbo.funcs, 100)

# Print the results
print("BBOB:")
print(f"Black Box Optimization using BBOB")
print(f"Score: {bbo.score}")

# Select the best solution
best_solution = bbo.funcs[np.argmax(bbo.score)]

# Print the best solution
print(f"\nBest Solution: {best_solution}")
print(f"Score: {bbo.score}")

# Update the algorithm with the best solution
bbo.funcs = [f(x) for x in best_solution]
bbo.evaluate_fitness(f, bbo.funcs, 100)

# Print the updated results
print("\nUpdated BBOB:")
print(f"Black Box Optimization using BBOB")
print(f"Score: {bbo.score}")

# Select the best solution
best_solution = bbo.funcs[np.argmax(bbo.score)]

# Print the best solution
print(f"\nBest Solution: {best_solution}")
print(f"Score: {bbo.score}")