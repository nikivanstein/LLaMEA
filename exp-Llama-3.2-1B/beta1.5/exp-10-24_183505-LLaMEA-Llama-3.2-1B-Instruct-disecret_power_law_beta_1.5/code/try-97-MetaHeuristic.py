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
        self.budget_history = []  # Initialize the budget history

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
            # Record the budget for the current evaluation
            self.budget_history.append(self.budget)
        # Return the best function found
        return self.best_func

    def set_budget(self, budget):
        self.budget = budget

    def get_best_func(self):
        return self.best_func

    def mutate(self, individual):
        # Randomly mutate the individual
        mutated_individual = individual.copy()
        # Randomly select a mutation point
        mutation_point = random.randint(0, self.dim)
        # Swap the mutation point with a random point in the search space
        mutated_individual[mutation_point], mutated_individual[mutation_point + 1] = mutated_individual[mutation_point + 1], mutated_individual[mutation_point]
        # Return the mutated individual
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
        self.budget_history = []  # Initialize the budget history
        self.mutation_history = []  # Initialize the mutation history

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
            # Record the budget for the current evaluation
            self.budget_history.append(self.budget)
            # Record the mutation history for the current evaluation
            self.mutation_history.append((point, random.uniform(self.search_space[0], self.search_space[1])))
        # Return the best function found
        return self.best_func

    def set_budget(self, budget):
        self.budget = budget

    def get_best_func(self):
        return self.best_func

    def mutate(self, individual):
        # Randomly mutate the individual
        mutated_individual = individual.copy()
        # Randomly select a mutation point
        mutation_point = random.randint(0, self.dim)
        # Swap the mutation point with a random point in the search space
        mutated_individual[mutation_point], mutated_individual[mutation_point + 1] = mutated_individual[mutation_point + 1], mutated_individual[mutation_point]
        # Return the mutated individual
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

# Refine the strategy by changing the mutation point
def mutate_point(point):
    return point[0] + random.uniform(-1, 1), point[1] + random.uniform(-1, 1)

meta_heuristic.mutation = mutate_point

# Optimize the test function using the MetaHeuristic
best_func = meta_heuristic(__call__, 100)

# Print the best function found
print("Best function:", best_func)
print("Best fitness:", best_func.budget)

# Plot the budget and mutation history
plt.plot(meta_heuristic.budget_history)
plt.plot(meta_heuristic.mutation_history[:, 0])
plt.xlabel("Budget")
plt.ylabel("Mutation Point")
plt.show()