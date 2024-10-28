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
            # Append the current iteration to the iteration history
            self.iteration_history.append((self.iterations, fitness, point, func))

            # If the current iteration is the maximum number of iterations, break the loop
            if self.iterations >= self.budget:
                break

        # Return the best function found
        return self.best_func

    def set_budget(self, budget):
        self.budget = budget

    def get_best_func(self):
        return self.best_func

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
            # Append the current iteration to the iteration history
            self.iteration_history.append((self.iterations, fitness, point, func))

            # If the current iteration is the maximum number of iterations, break the loop
            if self.iterations >= self.budget:
                break

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

# Refine the strategy by changing the individual lines of the selected solution
# Update the best function to use a different mutation operator
def mutate_func(point, mutation_operator):
    if mutation_operator =='swap':
        return (point[0] + random.uniform(-1, 1), point[1] + random.uniform(-1, 1))
    elif mutation_operator == 'crossover':
        return (point[0] + random.uniform(0, 1), point[1] + random.uniform(0, 1))
    else:
        raise ValueError("Invalid mutation operator")

# Update the MetaHeuristic instance
meta_heuristic.set_budget(100)
meta_heuristic.setMutationOperator('swap')

# Optimize the test function using the updated MetaHeuristic instance
best_func = meta_heuristic(__call__, 100)

# Print the best function found
print("Best function:", best_func)
print("Best fitness:", best_func.budget)

# Plot the iteration history
plt.plot(meta_heuristic.iteration_history)
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.show()