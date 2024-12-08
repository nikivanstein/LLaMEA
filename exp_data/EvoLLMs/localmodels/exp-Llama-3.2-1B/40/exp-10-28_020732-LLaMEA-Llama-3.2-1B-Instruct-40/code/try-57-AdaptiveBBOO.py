# Description: Adaptive Black Box Optimization using Evolutionary Strategies
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

    def optimize(self, func, population_size=100, mutation_rate=0.01, num_generations=100):
        # Initialize the search space
        x = self.search_space[np.random.choice(self.search_space.shape[0], self.budget)]

        # Initialize the best solution and best function value
        best_x = x
        best_func_value = func(x)

        # Perform the metaheuristic search
        for _ in range(num_generations):
            # Evaluate the function at the current search point
            func_value = func(x)

            # If the function value is better than the best found so far, update the best solution
            if func_value > best_func_value:
                best_x = x
                best_func_value = func_value

            # Perform population evolution
            for _ in range(population_size):
                # Randomly perturb the search point
                x = x + np.random.uniform(-1, 1, self.dim)

                # Evaluate the function at the new search point
                func_value = func(x)

                # If the function value is better than the best found so far, update the best solution
                if func_value > best_func_value:
                    best_x = x
                    best_func_value = func_value

            # Evaluate the best solution at the final search point
            func_value = func(best_x)

            # Print the best solution
            print(f"Best solution: {best_x}, Best function value: {func_value}")

        return best_x, func_value

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBOO(1000, 10)  # Increased budget to 1000

# Optimize the function with 1000 evaluations
bboo.optimize(func)

# Plot the results
plt.plot(bboo.func_evaluations)
plt.xlabel("Evaluations")
plt.ylabel("Function Value")
plt.title("Adaptive Black Box Optimization Results")
plt.show()

# One-line description with the main idea
# "Adaptive Black Box Optimization using Evolutionary Strategies"
# to optimize complex black box functions with a wide range of tasks and a high degree of adaptability"