# Description: AdaptiveBBOO: A novel metaheuristic algorithm for black box optimization tasks.
# Code: 
# ```python
import numpy as np
import matplotlib.pyplot as plt
import random

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = []
        self.budgets = []

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

        return best_x, func_value

    def select_strategy(self, x, func_value):
        # Select a strategy based on the number of evaluations
        if self.func_evaluations[0].count(x) > 0:
            return "Random"
        elif np.all(x >= self.search_space):
            return "Optimize"
        else:
            return "Refine"

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBO()

# Optimize the function with 1000 evaluations
bboo.optimize(func)

# Define a strategy to refine the solution
def refine_strategy(x, func_value):
    if np.all(x >= self.search_space):
        return "Optimize"
    elif self.func_evaluations[0].count(x) > 0:
        return "Random"
    else:
        return "Refine"

# Create an instance of the AdaptiveBBOO class with a refined strategy
bboo = AdaptiveBBOO(bboo.budget, 5)

# Optimize the function with 1000 evaluations
bboo.optimize(func)

# Define a strategy to refine the solution
def refine_strategy_bboo(bboo, func):
    def refine(x):
        return refine_strategy(x, func(x))

    return refine

# Create an instance of the AdaptiveBBOO class with a refined strategy
bboo = AdaptiveBBOO(bboo.budget, 5)
bboo.refine_strategy = refine_strategy_bboo(bboo, func)

# Optimize the function with 1000 evaluations
bboo.optimize(func)

# Plot the results
x_values = np.linspace(-5.0, 5.0, 100)
func_values = [func(x) for x in x_values]
best_x_values = [x for x, func_value in zip(x_values, func_values) if func_value > max(func_values)]
best_func_values = [func_value for x, func_value in zip(x_values, func_values) if func_value > max(func_values)]

plt.plot(x_values, func_values, label='Black Box')
plt.plot(x_values, best_x_values, label='Refined')
plt.plot(x_values, best_func_values, label='Optimized')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Black Box Optimization')
plt.legend()
plt.show()