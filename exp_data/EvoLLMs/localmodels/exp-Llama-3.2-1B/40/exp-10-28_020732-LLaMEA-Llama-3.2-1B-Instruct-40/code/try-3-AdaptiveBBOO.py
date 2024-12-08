# Description: AdaptiveBBOO: Adaptive Black Box Optimization using Adaptive Search Space and Mutation
# Code: 
# ```python
import numpy as np
import matplotlib.pyplot as plt

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

    def optimize(self, func, mutation_rate, mutation_threshold, epsilon, alpha, beta):
        # Initialize the search space
        x = self.search_space[np.random.choice(self.search_space.shape[0], self.budget)]

        # Perform the first evaluation
        func_value = func(x)
        print(f"Initial evaluation: {func_value}")

        # Initialize the best solution and best function value
        best_x = x
        best_func_value = func_value

        # Initialize the population
        population = [x.copy() for x in self.func_evaluations]

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

            # Perform mutation
            if np.random.rand() < mutation_rate:
                # Randomly select an individual to mutate
                new_individual = population[np.random.choice(len(population))]

                # Randomly select a mutation point
                mutation_point = np.random.randint(0, len(new_individual))

                # Apply mutation
                new_individual[mutation_point] += np.random.uniform(-epsilon, epsilon)
                new_individual[mutation_point] = max(-mutation_threshold, min(new_individual[mutation_point], mutation_threshold))

            # Update the population
            population = [x.copy() for x in self.func_evaluations]

            # Evaluate the best solution at the final search point
            func_value = func(best_x)

            # If the search space is exhausted, stop the algorithm
            if np.all(x >= self.search_space):
                break

            # Update the best solution and best function value
            best_x = x
            best_func_value = func_value

        # Evaluate the best solution at the final search point
        func_value = func(best_x)

        print(f"Best solution: {best_x}, Best function value: {func_value}")

        return best_x, func_value

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Define the parameters for the adaptive search
budget = 1000
dim = 5
mutation_rate = 0.01
mutation_threshold = 1.0
epsilon = 0.1
alpha = 0.5
beta = 0.1

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBOO(budget, dim)

# Optimize the function with the adaptive search
bboo.optimize(func, mutation_rate, mutation_threshold, epsilon, alpha, beta)