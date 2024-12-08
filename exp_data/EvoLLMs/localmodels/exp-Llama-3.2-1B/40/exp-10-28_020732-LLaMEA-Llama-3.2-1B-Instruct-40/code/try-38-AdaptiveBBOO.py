import numpy as np
import matplotlib.pyplot as plt

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = []
        self.best_individual = None
        self.best_function_value = np.inf

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

        # Update the best solution and function value if necessary
        if func_value < self.best_function_value:
            self.best_individual = best_x
            self.best_function_value = func_value

        # Print the best solution and function value
        print(f"Best solution: {best_x}, Best function value: {func_value}")

        return best_x, func_value

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Define the AdaptiveBBOO algorithm
adaptive_bboo = AdaptiveBBOO(1000, 10)

# Optimize the function
adaptive_bboo.optimize(func)

# Print the final best solution and function value
print(f"Final best solution: {adaptive_bboo.best_individual}, Final best function value: {adaptive_bboo.best_function_value}")

# Plot the fitness landscape
plt.figure(figsize=(8, 6))
for i in range(10):
    x = np.linspace(-5.0, 5.0, 100)
    y = func(x)
    plt.plot(x, y, label=f"Individual {i+1}")

plt.xlabel("Individual Index")
plt.ylabel("Fitness Value")
plt.title("Fitness Landscape")
plt.legend()
plt.show()