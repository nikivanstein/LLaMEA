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

    def optimize(self, func, population_size=100, mutation_rate=0.01, refit_rate=0.4):
        # Initialize the search space
        x = self.search_space[np.random.choice(self.search_space.shape[0], self.budget)]

        # Perform the first evaluation
        func_value = func(x)
        print(f"Initial evaluation: {func_value}")

        # Initialize the best solution and best function value
        best_x = x
        best_func_value = func_value

        # Initialize the population with random individuals
        population = [x for _ in range(population_size)]

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

            # Refit the best solution if the mutation rate is met
            if np.random.rand() < refit_rate:
                x = self.refit_solution(x, func_value)

            # Perform mutation on the best solution
            if np.random.rand() < mutation_rate:
                x = self.mutate_solution(x, func_value)

        # Evaluate the best solution at the final search point
        func_value = func(best_x)

        print(f"Best solution: {best_x}, Best function value: {func_value}")

        return best_x, func_value

    def refit_solution(self, x, func_value):
        # Refit the best solution by evaluating the function at multiple points
        # and selecting the one with the best function value
        best_x = x
        best_func_value = func_value
        for i in range(10):
            x = x + np.random.uniform(-1, 1, self.dim)
            func_value = func(x)
            if func_value > best_func_value:
                best_x = x
                best_func_value = func_value
        return best_x

    def mutate_solution(self, x, func_value):
        # Mutate the best solution by randomly changing one element
        # with a probability of mutation_rate
        if np.random.rand() < mutation_rate:
            x[np.random.randint(0, self.dim)] = np.random.uniform(-5.0, 5.0)
        return x

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBOO(1000, 10)

# Optimize the function with 1000 evaluations
bboo.optimize(func)