import numpy as np
import matplotlib.pyplot as plt
import random

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = []

    def __call__(self, func):
        def func_wrapper(x):
            return func(x)
        self.func_evaluations.append(func_wrapper)
        return func_wrapper

    def optimize(self, func, population_size=100, mutation_rate=0.01):
        # Initialize the search space
        x = self.search_space[np.random.choice(self.search_space.shape[0], self.budget)]

        # Perform the first evaluation
        func_value = func(x)
        print(f"Initial evaluation: {func_value}")

        # Initialize the best solution and best function value
        best_x = x
        best_func_value = func_value

        # Perform the metaheuristic search
        for _ in range(self.budget):
            # Evaluate the function at the current search point
            func_value = func(x)

            # If the function value is better than the best found so far, update the best solution
            if func_value > best_func_value:
                best_x = x
                best_func_value = func_value

            # If the search space is exhausted, stop the algorithm
            if np.all(x >= self.search_space):
                break

            # Randomly perturb the search point
            x = x + np.random.uniform(-1, 1, self.dim)

        # Evaluate the best solution at the final search point
        func_value = func(best_x)

        print(f"Best solution: {best_x}, Best function value: {func_value}")

        return best_x, func_value

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Define a population-based optimization algorithm
class PopulationBBOO:
    def __init__(self, budget, dim, population_size=100, mutation_rate=0.01):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate

    def __call__(self, func):
        # Initialize the population
        self.population = [func(np.random.uniform(-5.0, 5.0, self.dim)) for _ in range(self.population_size)]

        # Perform the metaheuristic search
        for _ in range(self.budget):
            # Evaluate the function at the current population
            func_values = [func(x) for x in self.population]
            func_values.sort()

            # Select the best solution
            best_index = np.argmax(func_values)
            best_x = self.population[best_index]

            # Perform mutation
            if random.random() < self.mutation_rate:
                mutation_index = random.randint(0, self.population_size - 1)
                self.population[mutation_index] = func(best_x)

        # Evaluate the best solution at the final population
        best_x = self.population[0]
        best_func_value = func(best_x)

        print(f"Best solution: {best_x}, Best function value: {best_func_value}")

        return best_x, best_func_value

# Define a simple population-based optimization algorithm
bboo = PopulationBBOO(budget=1000, dim=5, mutation_rate=0.01)

# Optimize the function with 1000 evaluations
bboo.optimize(func)

# Update the AdaptiveBBOO algorithm with the new solution
bboo = AdaptiveBBOO(budget=1000, dim=5)

# Optimize the function with 1000 evaluations
bboo.optimize(func)