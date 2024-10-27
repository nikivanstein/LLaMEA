import numpy as np
import random

class HyperCanny:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Grid search
            x_values = np.linspace(-5.0, 5.0, 100)
            y_values = func(x_values)
            grid = dict(zip(x_values, y_values))
            best_x, best_y = None, None
            for x, y in grid.items():
                if x < best_x or (x == best_x and y < best_y):
                    best_x, best_y = x, y
            # Random search
            random_x_values = np.random.uniform(-5.0, 5.0, self.dim)
            random_y_values = func(random_x_values)
            random_x_values = np.array([x for x, y in zip(random_x_values, random_y_values) if -5.0 <= x <= 5.0])
            random_y_values = np.array([y for x, y in zip(random_x_values, random_y_values) if -5.0 <= y <= 5.0])
            # Evolutionary algorithm
            self.func_evals += 1
            x_values = random_x_values
            y_values = random_y_values
            for _ in range(100):
                x_new = x_values + np.random.uniform(-0.1, 0.1, self.dim)
                y_new = y_values + np.random.uniform(-0.1, 0.1, self.dim)
                if -5.0 <= x_new <= 5.0 and -5.0 <= y_new <= 5.0:
                    x_values = x_new
                    y_values = y_new
                    break
            # Check if the new solution is better
            if np.max(y_values) > np.max(y_values + 0.1):
                best_x, best_y = x_values, y_values
        return best_x, best_y

    def mutation(self, individual, mutation_rate):
        if np.random.rand() < mutation_rate:
            index = np.random.randint(0, self.dim)
            individual[index], individual[index + 1] = individual[index + 1], individual[index]
        return individual

# One-line description:
# HyperCanny: A novel metaheuristic algorithm for solving black box optimization problems using a combination of grid search, random search, and evolutionary algorithms.

# Evaluation code:
import numpy as np

# Define the test functions
def func1(x):
    return np.sin(x)

def func2(x):
    return np.cos(x)

# Create an instance of HyperCanny
hc = HyperCanny(100, 10)

# Evaluate the test functions
func1_values = [hc(func1(x)) for x in np.linspace(-10, 10, 100)]
func2_values = [hc(func2(x)) for x in np.linspace(-10, 10, 100)]

# Print the results
print("Func1:")
print(func1_values)
print("Func2:")
print(func2_values)

# Evaluate the test functions using the HyperCanny algorithm
best_func1, best_func2 = hc(func1, func2)
print("\nBest solution for Func1:", best_func1)
print("Best solution for Func2:", best_func2)

# Print the fitness of the best solution
print("Fitness of best solution:", hc.evaluate_fitness(best_func1, hc.logger))
print("Fitness of best solution for Func2:", hc.evaluate_fitness(best_func2, hc.logger))