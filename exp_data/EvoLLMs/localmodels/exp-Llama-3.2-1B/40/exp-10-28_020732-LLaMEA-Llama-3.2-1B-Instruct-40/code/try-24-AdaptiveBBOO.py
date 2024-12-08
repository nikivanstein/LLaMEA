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

    def mutate(self, individual):
        # Randomly select an individual from the search space
        i = np.random.choice(self.search_space.shape[0])

        # Randomly perturb the individual
        x = individual + np.random.uniform(-1, 1, self.dim)

        # Check if the individual is within the search space
        if np.all(x >= self.search_space):
            return x
        else:
            return None

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Define a mutation function
def mutate_func(individual):
    return individual + np.random.uniform(-1, 1, individual.shape[0])

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBO()

# Optimize the function with 1000 evaluations
bboo.optimize(func)

# Create a new individual with a mutation
new_individual = mutate_func(bboo.search_space[0])

# Optimize the new individual
bboo.optimize(mutate_func(new_individual))

# Evaluate the best solution to update
bboo.func_evaluations.append(func)

# Update the best solution
best_x, best_func_value = bboo.optimize(func)
print(f"Best solution: {best_x}, Best function value: {best_func_value}")

# Plot the results
plt.plot(bboo.func_evaluations)
plt.xlabel('Evaluation')
plt.ylabel('Function Value')
plt.title('Black Box Optimization with Adaptive BBOO')
plt.show()