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
        self.strategy = "Random"  # Initialize the strategy

    def __call__(self, func, max_evals):
        # Evaluate the function up to max_evals times
        for _ in range(max_evals):
            # Randomly sample a point in the search space
            if self.strategy == "Random":
                point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            elif self.strategy == "LineSearch":
                # Apply line search to find the optimal point
                point = self.line_search(point)
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

    def line_search(self, point):
        # Apply line search to find the optimal point
        # Calculate the step size
        step_size = 0.01
        # Calculate the new point
        new_point = (point[0] + step_size, point[1])
        # Evaluate the function at the new point
        fitness = func(new_point)
        # Return the new point
        return new_point

    def set_strategy(self, strategy):
        self.strategy = strategy

# Description: "MetaHeuristics for Black Box Optimization" with a refined strategy
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
        self.strategy = "Random"  # Initialize the strategy
        self.strategy_params = {"epsilon": 0.03, "max_iter": 100}  # Initialize the strategy parameters

    def __call__(self, func, max_evals):
        # Evaluate the function up to max_evals times
        for _ in range(max_evals):
            # Randomly sample a point in the search space
            if self.strategy == "Random":
                point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            elif self.strategy == "LineSearch":
                # Apply line search to find the optimal point
                point = self.line_search(point)
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

    def line_search(self, point):
        # Apply line search to find the optimal point
        # Calculate the step size
        step_size = 0.01
        # Calculate the new point
        new_point = (point[0] + step_size, point[1])
        # Evaluate the function at the new point
        fitness = func(new_point)
        # Return the new point
        return new_point

    def set_strategy(self, strategy):
        self.strategy = strategy
        self.strategy_params = {"epsilon": 0.03, "max_iter": 100}  # Initialize the strategy parameters

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
meta_heuristic.set_strategy("LineSearch")
meta_heuristic.set_strategy_params({"epsilon": 0.03, "max_iter": 100})
best_func = meta_heuristic(__call__, 100)

# Print the best function found
print("Best function (refined):", best_func)
print("Best fitness (refined):", best_func.budget)

An exception occured: Traceback (most recent call last):
  File "/root/LLaMEA/llamea/llamea.py", line 187, in initialize_single
    new_individual = self.evaluate_fitness(new_individual)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/LLaMEA/mutation_exp.py", line 32, in evaluateBBOB
    exec(code, globals())
  File "<string>", line 96, in <module>
NameError: name '__call__' is not defined
.

# Description: "MetaHeuristics for Black Box Optimization" with a refined strategy
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
        self.strategy = "Random"  # Initialize the strategy
        self.strategy_params = {"epsilon": 0.03, "max_iter": 100}  # Initialize the strategy parameters
        self.strategy_history = []  # Initialize the strategy history

    def __call__(self, func, max_evals):
        # Evaluate the function up to max_evals times
        for _ in range(max_evals):
            # Randomly sample a point in the search space
            if self.strategy == "Random":
                point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            elif self.strategy == "LineSearch":
                # Apply line search to find the optimal point
                point = self.line_search(point)
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
            # Add the current point to the strategy history
            self.strategy_history.append((point, fitness))
        # Return the best function found
        return self.best_func

    def set_budget(self, budget):
        self.budget = budget

    def get_best_func(self):
        return self.best_func

    def line_search(self, point):
        # Apply line search to find the optimal point
        # Calculate the step size
        step_size = 0.01
        # Calculate the new point
        new_point = (point[0] + step_size, point[1])
        # Evaluate the function at the new point
        fitness = func(new_point)
        # Return the new point
        return new_point

    def set_strategy(self, strategy):
        self.strategy = strategy
        self.strategy_params = {"epsilon": 0.03, "max_iter": 100}  # Initialize the strategy parameters

    def set_strategy_history(self, strategy_history):
        self.strategy_history = strategy_history

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
meta_heuristic.set_strategy("LineSearch")
meta_heuristic.set_strategy_params({"epsilon": 0.03, "max_iter": 100})
meta_heuristic.set_strategy_history([((1.0, 2.0), 1.0), ((2.0, 3.0), 2.0), ((3.0, 4.0), 3.0)])
best_func = meta_heuristic(__call__, 100)

# Print the best function found
print("Best function (refined):", best_func)
print("Best fitness (refined):", best_func.budget)

# Print the strategy history
print("Strategy history:", meta_heuristic.strategy_history)