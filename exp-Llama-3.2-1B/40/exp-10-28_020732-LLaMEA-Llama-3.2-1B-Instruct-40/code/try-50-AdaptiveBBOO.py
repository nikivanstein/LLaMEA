# Description: AdaptiveBBOO: An adaptive black box optimization algorithm using a novel metaheuristic approach.
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

    def optimize(self, func):
        # Initialize the search space
        x = self.search_space[np.random.choice(self.search_space.shape[0], self.budget)]

        # Perform the first evaluation
        func_value = func(x)
        print(f"Initial evaluation: {func_value}")

        # Initialize the best solution and best function value
        best_x = x
        best_func_value = func_value

        # Initialize the population of solutions
        self.population = [x for x in np.random.uniform(self.search_space, self.search_space + 1, self.budget)]

        # Define the mutation function
        def mutate(x):
            return x + np.random.uniform(-1, 1, self.dim)

        # Define the selection function
        def select(x):
            return x[np.random.choice(self.budget)]

        # Define the crossover function
        def crossover(x1, x2):
            return np.clip(x1 + np.random.uniform(-1, 1, self.dim), -5.0, 5.0) + np.clip(x2 + np.random.uniform(-1, 1, self.dim), -5.0, 5.0)

        # Define the fitness function
        def fitness(x):
            return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

        # Perform the metaheuristic search
        for _ in range(self.budget):
            # Evaluate the function at the current search point
            func_value = fitness(x)

            # If the function value is better than the best found so far, update the best solution
            if func_value > best_func_value:
                best_x = x
                best_func_value = func_value

            # If the search space is exhausted, stop the algorithm
            if np.all(x >= self.search_space):
                break

            # Randomly select a parent from the population
            parent = select(self.population[np.random.randint(0, len(self.population))])

            # Perform crossover and mutation
            child = crossover(parent, x)
            child = mutate(child)

            # Evaluate the child function
            child_value = fitness(child)

            # If the child function value is better than the best found so far, update the best solution
            if child_value > best_func_value:
                best_x = child
                best_func_value = child_value

            # Update the population
            self.population = [child for child in np.random.uniform(self.search_space, self.search_space + 1, self.budget) if np.all(child >= self.search_space)]

        # Evaluate the best solution at the final search point
        func_value = fitness(best_x)

        print(f"Best solution: {best_x}, Best function value: {func_value}")

        return best_x, func_value

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBOO(1000, 10)

# Optimize the function
bboo.optimize(func)

# Plot the fitness values
plt.plot(bboo.population, bboo.func_evaluations)
plt.xlabel('Individual Index')
plt.ylabel('Fitness Value')
plt.title('Fitness Values Over Time')
plt.show()