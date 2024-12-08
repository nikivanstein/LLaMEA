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

    def adaptive_line_search(self, x, func, alpha=0.4):
        if np.all(x >= self.search_space):
            return x
        step = self.search_space[np.random.choice(self.search_space.shape[0], 1)][np.random.choice(self.search_space.shape[0], 1)]
        return x + alpha * (func(x + step) - func(x)) * step

    def __call__(self, func):
        def func_wrapper(x):
            return func(x)
        self.func_evaluations.append(func_wrapper)
        return func_wrapper

    def optimize(self, func, initial_x, iterations=1000):
        # Initialize the search space
        x = initial_x

        # Perform the adaptive line search to refine the solution
        for _ in range(iterations):
            # Evaluate the function at the current search point
            func_value = func(x)

            # If the function value is better than the best found so far, update the best solution
            if func_value > np.max([func_value - 0.1 * (x - x) for x in self.func_evaluations]):
                x = self.adaptive_line_search(x, func)

        # Evaluate the best solution at the final search point
        func_value = func(x)

        print(f"Best solution: {x}, Best function value: {func_value}")

        return x, func_value

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBOO(1000, 10)

# Optimize the function with 1000 evaluations
bboo.optimize(func)

# Plot the results
plt.plot(bboo.func_evaluations)
plt.xlabel("Evaluation Iterations")
plt.ylabel("Best Function Value")
plt.title("Adaptive BBOO Optimization")
plt.show()