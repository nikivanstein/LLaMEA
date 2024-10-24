# Description: "MetaHeuristics for Black Box Optimization"
# Code: 
# ```python
import random
import numpy as np
from scipy.optimize import minimize

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
            # Store the current iteration
            self.iteration_history.append(self.iterations)
        # Return the best function found
        return self.best_func

    def set_budget(self, budget):
        self.budget = budget

    def get_best_func(self):
        return self.best_func

    def mutate(self, individual):
        # Randomly change one bit in the individual
        bit_index = random.randint(0, self.dim)
        individual[bit_index] = 1 - individual[bit_index]
        # Store the mutated individual
        self.iteration_history.append(self.iterations + 1)
        # Return the mutated individual
        return individual

    def crossover(self, parent1, parent2):
        # Select a random crossover point
        crossover_point = random.randint(0, self.dim)
        # Create the child individual
        child = parent1[:crossover_point] + parent2[crossover_point:]
        # Return the child individual
        return child

# Description: "MetaHeuristics for Black Box Optimization"
# Code: 
# ```python
# import random
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
        self.iteration_history = []  # Initialize the iteration history
        self.iteration_history_best = []  # Initialize the iteration history best

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
            # Store the current iteration
            self.iteration_history.append(self.iterations)
            # Store the current iteration history
            self.iteration_history_best.append(self.iterations)
        # Return the best function found
        return self.best_func

    def set_budget(self, budget):
        self.budget = budget

    def get_best_func(self):
        return self.best_func

    def mutate(self, individual):
        # Randomly change one bit in the individual
        bit_index = random.randint(0, self.dim)
        individual[bit_index] = 1 - individual[bit_index]
        # Store the mutated individual
        self.iteration_history.append(self.iterations + 1)
        # Return the mutated individual
        return individual

    def crossover(self, parent1, parent2):
        # Select a random crossover point
        crossover_point = random.randint(0, self.dim)
        # Create the child individual
        child = parent1[:crossover_point] + parent2[crossover_point:]
        # Return the child individual
        return child

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

# Print the iteration history
print("Iteration History:")
for i, iteration in enumerate(meta_heuristic.iteration_history):
    print(f"Iteration {i+1}: {iteration}")
print("Iteration History Best:")
for i, iteration in enumerate(meta_heuristic.iteration_history_best):
    print(f"Iteration {i+1}: {iteration}")