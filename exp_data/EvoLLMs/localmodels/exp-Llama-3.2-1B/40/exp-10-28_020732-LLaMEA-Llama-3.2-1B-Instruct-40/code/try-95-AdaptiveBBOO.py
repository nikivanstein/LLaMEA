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

        # Update the population with the best solution and its fitness
        self.population.append((best_x, func_value))

        # If the budget is reached, evaluate the population to select the next generation
        if len(self.population) == self.budget:
            # Evaluate the population at 1/4 of the budget
            self.population = [self.func_wrapper(x) for x, _ in self.population[:int(self.budget/4)]]
            # Evaluate the population at 1/4 of the budget
            self.population = [self.func_wrapper(x) for x, _ in self.population[int(self.budget/4):]]

        return best_x, func_value

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBOO(1000, 10)

# Optimize the function with 1000 evaluations
bboo.optimize(func)

# Plot the results
plt.plot(bboo.population[0][1], label='Initial solution')
plt.plot(bboo.population[1][1], label='Optimized solution')
plt.xlabel('Fitness')
plt.ylabel('x')
plt.title('Black Box Optimization using Evolutionary Strategies')
plt.legend()
plt.show()