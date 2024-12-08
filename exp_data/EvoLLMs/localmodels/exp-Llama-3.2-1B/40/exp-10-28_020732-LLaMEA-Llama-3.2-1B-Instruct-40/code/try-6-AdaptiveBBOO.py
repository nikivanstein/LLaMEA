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

        # Create a new individual based on the best solution
        new_individual = func(best_x)

        # Refine the strategy by changing the individual lines of the selected solution
        if np.random.rand() < 0.4:
            new_individual = self.refine_individual(new_individual)

        # Add the new individual to the population
        self.population.append(new_individual)

        # Evaluate the new individual
        new_func_value = func(new_individual)

        # Update the best individual and best function value
        best_x, best_func_value = self.evaluate_best_individual_and_function(best_x, new_func_value)

        # Update the population
        self.population = self.population[:self.budget]

        return best_x, best_func_value

    def evaluate_best_individual_and_function(self, best_x, new_func_value):
        best_individual = best_x
        best_function_value = new_func_value

        for individual, func in zip(self.population, self.func_evaluations):
            func_value = func(individual)
            if func_value > best_function_value:
                best_individual = individual
                best_function_value = func_value

        return best_individual, best_function_value

    def refine_individual(self, individual):
        # Randomly select a subset of dimensions to refine
        dim_subset = np.random.choice(self.dim, self.dim - len(self.refine_individual))

        # Refine the individual by changing the selected dimensions
        refined_individual = individual[:dim_subset] + np.random.uniform(-1, 1, dim_subset)

        return refined_individual

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBOO(1000, 5)

# Optimize the function with 1000 evaluations
bboo.optimize(func)

# Plot the results
plt.plot(bboo.population, bboo.func_evaluations)
plt.xlabel("Iteration")
plt.ylabel("Function Value")
plt.title("Adaptive Evolutionary Algorithm with Adaptation and Refinement")
plt.show()