# Description: "Black Box Optimization using Adaptive Mutation and Crossover"
# Code: 
# ```python
import random
import numpy as np

class MetaHeuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]  # Define the search space
        self.best_func = None  # Initialize the best function found so far
        self.best_fitness = float('inf')  # Initialize the best fitness found so far
        self.iterations = 0  # Initialize the number of iterations
        self.mutation_rate = 0.02  # Adaptive mutation rate

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

    def mutate(self, individual):
        # Apply mutation to the individual
        if random.random() < self.mutation_rate:
            # Randomly select two random genes
            idx1, idx2 = random.sample(range(len(individual)), 2)
            # Swap the genes
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        # Return the mutated individual
        return individual

    def crossover(self, parent1, parent2):
        # Perform crossover between the two parents
        child = parent1[:len(parent1)//2] + parent2[len(parent2)//2:]
        return child

# Description: "Black Box Optimization using Adaptive Mutation and Crossover"
# Code: 
# ```python
# import random
# import numpy as np

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

# Perform mutation and crossover to refine the strategy
def mutate_and_crossover(individual):
    mutated_individual = meta_heuristic.mutate(individual)
    child = meta_heuristic.crossover(mutated_individual, individual)
    return child

# Evaluate the best function found so far
best_fitness = best_func.budget
print("Best fitness:", best_fitness)

# Optimize the best function found so far using mutation and crossover
best_func = mutate_and_crossover(best_func)

# Print the best function found after optimization
print("Best function after optimization:", best_func)
print("Best fitness after optimization:", best_func.budget)