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
            # Add the current iteration to the iteration history
            self.iteration_history.append((point, fitness, self.budget))

    def set_budget(self, budget):
        self.budget = budget

    def get_best_func(self):
        return self.best_func

    def mutate(self, individual):
        # Randomly mutate the individual
        mutated_individual = individual.copy()
        if random.random() < 0.02:
            # Change the individual's fitness
            mutated_individual[0] = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            mutated_individual[1] = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            mutated_individual[2] += 1
        return mutated_individual

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
            # Add the current iteration to the iteration history
            self.iteration_history.append((point, fitness, self.budget))

    def set_budget(self, budget):
        self.budget = budget

    def get_best_func(self):
        return self.best_func

    def mutate(self, individual):
        # Randomly mutate the individual
        mutated_individual = individual.copy()
        if random.random() < 0.02:
            # Change the individual's fitness
            mutated_individual[0] = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            mutated_individual[1] = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            mutated_individual[2] += 1
        return mutated_individual

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

# Refine the strategy by changing the mutation probability
meta_heuristic.iteration_history = []
meta_heuristic.mutate = lambda individual: individual if random.random() < 0.02 else individual.copy()

# Optimize the test function using the refined strategy
best_func = meta_heuristic(__call__, 100)

# Print the best function found
print("Best function after refinement:", best_func)
print("Best fitness after refinement:", best_func.budget)

# Plot the iteration history
plt.plot(meta_heuristic.iteration_history, x for x, _ in meta_heuristic.iteration_history)
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.title('Iteration History')
plt.show()