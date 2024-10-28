# Description: Adaptive Black Box Optimization using AdaptiveBBOO Algorithm
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
        self.population_size = 100

    def __call__(self, func):
        def func_wrapper(x):
            return func(x)
        self.func_evaluations.append(func_wrapper)
        return func_wrapper

    def optimize(self, func, initial_population, mutation_rate, population_size, num_generations):
        # Initialize the population
        population = initial_population

        # Initialize the best solution and best function value
        best_x = None
        best_func_value = float('-inf')

        # Perform the metaheuristic search
        for _ in range(num_generations):
            # Evaluate the function at the current population
            func_values = [func(x) for x in population]
            func_values = np.array(func_values)

            # Select the fittest individuals
            fittest_individuals = np.argsort(func_values)[::-1][:self.population_size]

            # Select the next generation
            next_generation = []
            for _ in range(self.population_size):
                # Select a random individual from the fittest individuals
                individual = fittest_individuals[np.random.choice(fittest_individuals.shape[0])]

                # Perform mutation
                mutated_individual = individual + np.random.uniform(-1, 1, self.dim)

                # Evaluate the mutated individual
                func_value = func(mutated_individual)

                # If the mutated individual is better than the best found so far, update the best solution
                if func_value > best_func_value:
                    best_x = mutated_individual
                    best_func_value = func_value

            # Add the best individual to the next generation
            next_generation.append(best_x)

            # Replace the worst individuals with the next generation
            population = next_generation[:self.population_size]

            # If the search space is exhausted, stop the algorithm
            if np.all(population >= self.search_space):
                break

        # Evaluate the best solution at the final search point
        func_value = func(best_x)

        print(f"Best solution: {best_x}, Best function value: {func_value}")

        return best_x, func_value

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBOO(1000, 10)

# Optimize the function with 1000 evaluations
bboo.optimize(func, np.linspace(-5.0, 5.0, 100), 0.1, 100, 100)

# Plot the results
plt.plot(bboo.search_space, bboo.func_evaluations)
plt.xlabel('Search Space')
plt.ylabel('Function Value')
plt.title('Black Box Optimization using AdaptiveBBOO Algorithm')
plt.show()