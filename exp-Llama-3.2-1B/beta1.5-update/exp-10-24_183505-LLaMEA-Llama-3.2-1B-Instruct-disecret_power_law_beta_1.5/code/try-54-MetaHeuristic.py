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
        self.population = []  # Initialize the population

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
        # Refine the strategy by changing the individual lines of the selected solution
        individual = list(individual)
        while len(individual) < self.dim:
            # Change the individual line with a random probability
            if random.random() < 0.02:
                individual.append(random.uniform(self.search_space[0], self.search_space[1]))
        # Return the mutated individual
        return tuple(individual)

    def crossover(self, parent1, parent2):
        # Perform crossover between two parents to create a child
        child = tuple()
        for i in range(self.dim):
            if random.random() < 0.5:
                child += tuple(parent1[i])
            else:
                child += tuple(parent2[i])
        # Return the child
        return child

    def evaluateBBOB(self, func, max_evals):
        # Optimize the test function using the MetaHeuristic
        best_func = None
        best_fitness = float('inf')
        for _ in range(max_evals):
            # Randomly sample a point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            fitness = func(point)
            # If the fitness is better than the current best fitness, update the best function and fitness
            if fitness < best_fitness:
                best_func = func
                best_fitness = fitness
            # If the fitness is equal to the current best fitness, update the best function if it has a lower budget
            elif fitness == best_fitness and self.budget < best_func.budget:
                best_func = func
                best_fitness = fitness
        # Return the best function found
        return best_func

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
        self.population = []  # Initialize the population
        self.population_size = 100  # Initialize the population size

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
        # Refine the strategy by changing the individual lines of the selected solution
        individual = list(individual)
        while len(individual) < self.dim:
            # Change the individual line with a random probability
            if random.random() < 0.02:
                individual.append(random.uniform(self.search_space[0], self.search_space[1]))
        # Return the mutated individual
        return tuple(individual)

    def crossover(self, parent1, parent2):
        # Perform crossover between two parents to create a child
        child = tuple()
        for i in range(self.dim):
            if random.random() < 0.5:
                child += tuple(parent1[i])
            else:
                child += tuple(parent2[i])
        # Return the child
        return child

    def evaluateBBOB(self, func, max_evals):
        # Optimize the test function using the MetaHeuristic
        best_func = None
        best_fitness = float('inf')
        for _ in range(max_evals):
            # Randomly sample a point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            fitness = func(point)
            # If the fitness is better than the current best fitness, update the best function and fitness
            if fitness < best_fitness:
                best_func = func
                best_fitness = fitness
            # If the fitness is equal to the current best fitness, update the best function if it has a lower budget
            elif fitness == best_fitness and self.budget < best_func.budget:
                best_func = func
                best_fitness = fitness
        # Return the best function found
        return best_func

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

# Create a new population
new_population = meta_heuristic.__call__(test_func, 1000)

# Print the new population
print("New population:")
for individual in new_population:
    print(individual)

# Evaluate the new population using the MetaHeuristic
new_best_func = meta_heuristic.evaluateBBOB(test_func, 1000)

# Print the new best function found
print("New best function:", new_best_func)
print("New best fitness:", new_best_func.budget)

# Plot the results
import matplotlib.pyplot as plt
plt.plot(meta_heuristic.best_fitness, label='Original')
plt.plot(new_best_func.budget, label='New')
plt.plot(meta_heuristic.best_fitness, label='Original')
plt.legend()
plt.show()