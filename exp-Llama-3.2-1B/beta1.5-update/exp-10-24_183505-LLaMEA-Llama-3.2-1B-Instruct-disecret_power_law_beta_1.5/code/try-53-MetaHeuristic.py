# Description: Novel Metaheuristic for Black Box Optimization
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
            # Update the budget history
            self.budget_history.append(self.budget)
        # Return the best function found
        return self.best_func

    def set_budget(self, budget):
        self.budget = budget

    def get_best_func(self):
        return self.best_func

    def mutate(self, func):
        # Mutate the function by changing a random parameter
        param = random.uniform(self.search_space[0], self.search_space[1])
        func = func(param)
        # Update the budget history
        self.budget_history.append(self.budget)
        return func

# Description: Novel Metaheuristic for Black Box Optimization
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

# Refine the strategy by changing the mutation function
def mutate_func(x):
    param = random.uniform(-5.0, 5.0)
    x = x + param
    return x

# Create an instance of the MetaHeuristic class with the new mutation function
meta_heuristic = MetaHeuristic(100, 10)
meta_heuristic.set_budget(100)
meta_heuristic.mutate = mutate_func

# Optimize the test function using the new MetaHeuristic
best_func = meta_heuristic(__call__, 100)

# Print the best function found
print("Best function:", best_func)