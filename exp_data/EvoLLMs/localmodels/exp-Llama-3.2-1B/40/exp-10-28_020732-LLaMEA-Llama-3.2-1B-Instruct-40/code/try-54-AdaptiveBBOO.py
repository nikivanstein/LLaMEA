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

    def optimize(self, func, mutation_rate=0.1, evolution_rate=0.01):
        # Initialize the search space
        x = self.search_space[np.random.choice(self.search_space.shape[0], self.budget)]

        # Initialize the best solution and best function value
        best_x = x
        best_func_value = func(x)

        # Perform the metaheuristic search
        for _ in range(self.budget):
            # Evaluate the function at the current search point
            func_value = func(x)

            # If the function value is better than the best found so far, update the best solution
            if func_value > best_func_value:
                best_x = x
                best_func_value = func_value

            # Randomly perturb the search point
            x = x + np.random.uniform(-1, 1, self.dim)

            # Perform evolutionary search
            if np.random.rand() < evolution_rate:
                # Select the next individual using evolutionary search
                new_individual = self.select_individual(x)
                # Apply mutation to the new individual
                new_individual = self.applyMutation(new_individual)
                # Evaluate the new individual at the function
                new_func_value = func(new_individual)

                # If the new function value is better than the best found so far, update the best solution
                if new_func_value > best_func_value:
                    best_x = new_individual
                    best_func_value = new_func_value

            # Randomly mutate the search point
            if np.random.rand() < mutation_rate:
                # Select a random individual from the search space
                individual = self.select_individual(self.search_space)
                # Randomly perturb the individual
                individual = individual + np.random.uniform(-1, 1, self.dim)

        # Evaluate the best solution at the final search point
        func_value = func(best_x)

        print(f"Best solution: {best_x}, Best function value: {func_value}")

        return best_x, func_value

    def select_individual(self, x):
        # Select the individual with the highest fitness value
        return x[np.argmax(self.func_evaluations functional_index(x))]

    def applyMutation(self, individual):
        # Apply mutation to the individual
        return individual + np.random.uniform(-1, 1, self.dim)

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBOO(1000)

# Optimize the function with 1000 evaluations
bboo.optimize(func)

# Plot the results
plt.plot(np.linspace(-5.0, 5.0, 100), bboo.func_evaluations)
plt.xlabel("Search Point")
plt.ylabel("Function Value")
plt.show()