# Description: "MetaHeuristics for Black Box Optimization"
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
        self.fitness_history = []  # Initialize the fitness history

    def __call__(self, func, max_evals):
        # Evaluate the function up to max_evals times
        for _ in range(max_evals):
            # Randomly sample a point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            fitness = func(point)
            # Append the fitness to the fitness history
            self.fitness_history.append(fitness)
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
best_func = meta_heuristic(test_func, 100)

# Print the best function found
print("Best function:", best_func)

# Plot the fitness history
plt.plot(meta_heuristic.fitness_history)
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.title('Fitness History')
plt.show()

# Refine the strategy
def refine_strategy(func, meta_heuristic, max_evals):
    for _ in range(max_evals):
        # Randomly sample a point in the search space
        point = (random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0))
        # Evaluate the function at the point
        fitness = func(point)
        # Append the fitness to the fitness history
        meta_heuristic.fitness_history.append(fitness)
        # If the fitness is better than the current best fitness, update the best function and fitness
        if fitness < meta_heuristic.best_fitness:
            meta_heuristic.best_func = func
            meta_heuristic.best_fitness = fitness
        # If the fitness is equal to the current best fitness, update the best function if it has a lower budget
        elif fitness == meta_heuristic.best_fitness and meta_heuristic.budget < meta_heuristic.best_func.budget:
            meta_heuristic.best_func = func
            meta_heuristic.best_fitness = fitness
    # Return the best function found
    return meta_heuristic.best_func

# Optimize the test function using the refined strategy
best_func = refine_strategy(test_func, meta_heuristic, 100)

# Print the best function found
print("Best function after refinement:", best_func)

# Plot the fitness history with the refined strategy
plt.plot(meta_heuristic.fitness_history)
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.title('Fitness History with Refinement')
plt.show()