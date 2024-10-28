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
        self.population_size = 100
        self.population = np.random.choice(self.search_space, self.population_size, replace=False)
        self.fitness_values = np.zeros(self.population_size)
        self.population_history = []

    def __call__(self, func):
        def func_wrapper(x):
            return func(x)
        self.func_evaluations.append(func_wrapper)
        return func_wrapper

    def mutate(self, individual):
        # Randomly select a dimension to mutate
        dim = np.random.choice(self.dim)
        # Perturb the individual in the selected dimension
        individual[dim] = np.random.uniform(-1, 1, self.dim)
        # Evaluate the new individual
        new_fitness_value = func(individual)
        # Update the fitness value of the original individual
        self.fitness_values[individual] = new_fitness_value
        return individual

    def evolve_population(self, num_generations):
        for _ in range(num_generations):
            # Select parents using tournament selection
            parents = np.array([self.population[np.random.choice(self.population_size, 2, replace=False)] for _ in range(self.population_size)])
            # Evaluate the fitness values of the parents
            parents_fitness_values = np.array([self.fitness_values[parent] for parent in parents])
            # Select the fittest parents
            parents = np.array([parent for _, parent in sorted(zip(parents_fitness_values, parents), reverse=True)])
            # Create a new population by crossover and mutation
            self.population = np.array([self.mutate(parent) for parent in parents])
            # Evaluate the fitness values of the new population
            self.fitness_values = np.array([self.fitness_values[parent] for parent in self.population])
            # Store the new population history
            self.population_history.append(self.population)

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

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBOO(1000, 10)

# Optimize the function with 1000 evaluations
bboo.optimize(func)

# Plot the population history
plt.figure(figsize=(10, 6))
plt.plot(bboo.population_history)
plt.title("Population History")
plt.xlabel("Generation")
plt.ylabel("Population Size")
plt.show()

# Plot the fitness values
plt.figure(figsize=(10, 6))
plt.plot(bboo.fitness_values)
plt.title("Fitness Values")
plt.xlabel("Generation")
plt.ylabel("Fitness Value")
plt.show()

# Plot the best solutions
plt.figure(figsize=(10, 6))
plt.plot(bboo.population_history[:, 0], label="Best Individual")
plt.plot(bboo.population_history[:, 1], label="Best Individual")
plt.title("Best Solutions")
plt.xlabel("Generation")
plt.ylabel("Individual")
plt.legend()
plt.show()