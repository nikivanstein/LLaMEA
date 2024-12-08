import numpy as np
import matplotlib.pyplot as plt

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = []
        self.probabilistic_line_search = False

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

            # Randomly perturb the search point with probabilistic line search
            if self.probabilistic_line_search:
                # Update the best solution using probabilistic line search
                best_x, best_func_value = self.probabilistic_line_search(x, func_value)

            # Randomly perturb the search point
            x = x + np.random.uniform(-1, 1, self.dim)

        # Evaluate the best solution at the final search point
        func_value = func(best_x)

        print(f"Best solution: {best_x}, Best function value: {func_value}")

        return best_x, func_value

    def probabilistic_line_search(self, x, func_value):
        # Initialize the search direction
        dx = np.random.uniform(-1, 1, self.dim)

        # Perform the line search
        alpha = 0.1
        for i in range(1, self.dim):
            # Update the search direction
            dx[i] += alpha * (func_value - x[i])

        # Update the best solution using probabilistic line search
        best_x = x
        best_func_value = func_value
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

            # Update the search direction using probabilistic line search
            x += np.random.uniform(-1, 1, self.dim)
            dx = np.random.uniform(-1, 1, self.dim)

            # Perform the line search
            alpha = 0.1
            for i in range(1, self.dim):
                # Update the search direction
                dx[i] += alpha * (func_value - x[i])

            # Update the best solution using probabilistic line search
            best_x = x
            best_func_value = func_value

        return best_x, best_func_value

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBOO(1000, 10)

# Optimize the function with 1000 evaluations
bboo.optimize(func)

# Plot the best solution and function value
plt.plot(bboo.search_space, bboo.best_func_value)
plt.scatter(bboo.search_space[np.random.choice(bboo.search_space.shape[0], 10)], bboo.best_func_value)
plt.show()