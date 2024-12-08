import numpy as np
import matplotlib.pyplot as plt

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = []
        self.population = []

    def __call__(self, func):
        def func_wrapper(x):
            return func(x)
        self.func_evaluations.append(func_wrapper)
        return func_wrapper

    def optimize(self, func):
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

        # Refine the solution using adaptive search
        for _ in range(self.budget):
            # Select the next individual based on the budget
            if _ % 2 == 0:
                # Select the best solution so far
                best_x = self.population[np.argmax(self.population[::2])]

            # Select a random individual from the remaining population
            next_x = self.population[np.random.choice(self.population.shape[0], 1)[0]]

            # Perform a mutation on the next individual
            if np.random.rand() < 0.4:
                next_x += np.random.uniform(-1, 1, self.dim)

            # Evaluate the function at the next individual
            func_value = func(next_x)

            # If the function value is better than the best found so far, update the best solution
            if func_value > best_func_value:
                best_x = next_x
                best_func_value = func_value

        # Evaluate the best solution at the final search point
        func_value = func(best_x)

        print(f"Best solution: {best_x}, Best function value: {func_value}")

        # Update the population
        self.population = [best_x, next_x]

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBOO(1000, 10)

# Optimize the function with 1000 evaluations
bboo.optimize(func)

# Plot the results
plt.plot(np.linspace(-5, 5, 100), [func(x) for x in np.linspace(-5, 5, 100)])
plt.show()