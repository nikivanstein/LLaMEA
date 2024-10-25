# Description: "MetaHeuristics for Black Box Optimization"
# Code: 
# ```python
import random
import numpy as np
from scipy.optimize import differential_evolution

class MetaHeuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]  # Define the search space
        self.best_func = None  # Initialize the best function found so far
        self.best_fitness = float('inf')  # Initialize the best fitness found so far
        self.iterations = 0  # Initialize the number of iterations

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

# Description: "MetaHeuristics for Black Box Optimization"
# Code: 
# ```python
# import numpy as np
# import matplotlib.pyplot as plt

# Define the MetaHeuristic class
class MetaHeuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]  # Define the search space
        self.best_func = None  # Initialize the best function found so far
        self.best_fitness = float('inf')  # Initialize the best fitness found so far
        self.iterations = 0  # Initialize the number of iterations
        self.iteration_history = []  # Initialize the iteration history

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
            # Record the current iteration
            self.iteration_history.append((point, fitness))
        # Return the best function found
        return self.best_func

    def set_budget(self, budget):
        self.budget = budget

    def get_best_func(self):
        return self.best_func

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

# Refine the strategy using mutation and selection
def mutation(individual, mutation_rate):
    # Randomly select a mutation point
    mutation_point = random.randint(0, len(individual) - 1)
    # Swap the mutation point with a random point in the search space
    individual[mutation_point], individual[mutation_point] = individual[mutation_point], individual[mutation_point]
    # Update the iteration history
    self.iteration_history.append((individual, fitness(individual)))
    # Update the best function if it has a lower budget
    if self.budget < fitness(individual):
        self.best_func = individual
        self.best_fitness = fitness(individual)
    # Record the current iteration
    self.iteration_history.append((individual, fitness(individual)))
    # Return the mutated individual
    return individual

def selection(individual, num_individuals):
    # Record the current iteration
    self.iteration_history.append((individual, fitness(individual)))
    # Select the individual with the best fitness
    selected_individual = individual[np.argmax([fitness(individual) for fitness in self.iteration_history])]
    # Update the best function if it has a lower budget
    if self.budget < fitness(selected_individual):
        self.best_func = selected_individual
        self.best_fitness = fitness(selected_individual)
    # Return the selected individual
    return selected_individual

# Refine the strategy
mutation_rate = 0.02
selection_rate = 0.1

meta_heuristic.set_budget(100)
meta_heuristic.set_iteration_rate(0.1)
meta_heuristic.set_selection_rate(0.1)

best_func = meta_heuristic(__call__, 100)
print("Refined best function:", best_func)
print("Refined best fitness:", best_func.budget)