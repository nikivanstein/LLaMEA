# Description: "Black Box Optimization using Evolutionary Strategies"
# Code: 
# ```python
import random
import numpy as np
import matplotlib.pyplot as plt

class MetaHeuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]  # Define the search space
        self.best_func = None  # Initialize the best function found so far
        self.best_fitness = float('inf')  # Initialize the best fitness found so far
        self.iterations = 0  # Initialize the number of iterations
        self.population_size = 100  # Population size for selection
        self.mutation_rate = 0.01  # Mutation rate for selection

    def __call__(self, func, max_evals):
        # Evaluate the function up to max_evals times
        for _ in range(max_evals):
            # Randomly sample a point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            fitness = func(point)
            # If the fitness is better than the current best fitness, update the best function and fitness
            if fitness < self.best_fitness:
                self.best_func = func
                self.best_fitness = fitness
            # If the fitness is equal to the current best fitness, update the best function if it has a lower budget
            elif fitness == self.best_fitness and self.budget < self.best_func.budget:
                self.best_func = func
                self.best_fitness = fitness
        # Return the best function found
        return self.best_func

    def set_budget(self, budget):
        self.budget = budget

    def get_best_func(self):
        return self.best_func

    def select(self, func, max_evals):
        # Select the best individual from the population
        population = np.random.rand(self.population_size, self.dim)
        fitnesses = np.array([func(point) for point in population])
        selected = np.argsort(fitnesses)[:self.population_size // 2]
        return selected, fitnesses[selected]

    def mutate(self, selected, mutation_rate):
        # Mutate the selected individuals
        mutated = []
        for i in range(self.population_size):
            if random.random() < mutation_rate:
                mutated.append(selected[i])
            else:
                mutated.append(np.random.uniform(self.search_space[0], self.search_space[1]))
        return mutated

# Define a noiseless function
def noiseless_func(x):
    return np.sin(x)

# Define a noise function
def noise_func(x):
    return np.random.normal(0, 1, x)

# Define a test function
def test_func(x):
    return x**2 + 2*x + 1

# Create an instance of the MetaHeuristic class
meta_heuristic = MetaHeuristic(100, 10)

# Set the budget for the MetaHeuristic
meta_heuristic.set_budget(100)

# Optimize the test function using the MetaHeuristic
best_func = meta_heuristic(__call__, 100)

# Print the best function found
print("Best function:", best_func)
print("Best fitness:", best_func.budget)

# Refine the strategy
def refine_strategy(func, max_evals, mutation_rate):
    # Select the best individual from the population
    selected, fitnesses = meta_heuristic.select(func, max_evals)
    # Mutate the selected individuals
    mutated = meta_heuristic.mutate(selected, mutation_rate)
    # Evaluate the fitness of the mutated individuals
    fitnesses_mutated = np.array([func(point) for point in mutated])
    # Return the best individual and fitness
    return selected[np.argmax(fitnesses_mutated)], fitnesses_mutated

# Optimize the test function using the refined strategy
best_func_mutated, fitnesses_mutated = refine_strategy(test_func, 100, 0.01)

# Print the best mutated function found
print("Best mutated function:", best_func_mutated)
print("Best mutated fitness:", fitnesses_mutated)

# Plot the results
plt.plot(fitnesses, label='Original')
plt.plot(fitnesses_mutated, label='Refined')
plt.legend()
plt.show()