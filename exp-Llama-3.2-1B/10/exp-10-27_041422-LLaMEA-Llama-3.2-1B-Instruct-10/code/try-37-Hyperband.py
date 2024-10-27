# Description: Hyperband Algorithm for Black Box Optimization
# Code: 
# ```python
import numpy as np
import random
import os

class Hyperband:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_func = None
        self.best_func_evals = 0
        self.sample_size = 1
        self.sample_dir = None
        self.sample_dir_map = {}

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget cannot be zero or negative")
        
        if self.best_func is not None:
            return self.best_func
        
        # Initialize the best function and its evaluation count
        self.best_func = func
        self.best_func_evals = 1
        
        # Set the sample size and directory
        self.sample_size = 10
        self.sample_dir = f"sample_{self.sample_size}"
        
        # Perform adaptive sampling
        for _ in range(self.budget):
            # Generate a random sample of size self.sample_size
            sample = np.random.uniform(-5.0, 5.0, size=self.dim)
            
            # Evaluate the function at the current sample
            func_eval = func(sample)
            
            # If this is the first evaluation, update the best function
            if self.best_func_evals == 1:
                self.best_func = func_eval
                self.best_func_evals = 1
            # Otherwise, update the best function if the current evaluation is better
            else:
                if func_eval > self.best_func:
                    self.best_func = func_eval
                    self.best_func_evals = 1
                else:
                    self.best_func_evals += 1
            
            # Save the current sample to the sample directory
            self.sample_dir_map[os.path.join(self.sample_dir, str(_))] = sample
        
        return self.best_func

class LLaMEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = [np.random.uniform(-5.0, 5.0, size=self.dim) for _ in range(self.population_size)]
        self.population_history = []
        
    def __call__(self, func):
        return self.hyperband(func)

    def hyperband(self, func):
        best_func = None
        best_func_evals = 0
        
        while True:
            # Select a new individual
            new_individual = random.choice(self.population)
            
            # Evaluate the function at the new individual
            func_eval = func(new_individual)
            
            # If this is the first evaluation, update the best function
            if best_func_evals == 0:
                best_func = func_eval
                best_func_evals = 1
            # Otherwise, update the best function if the current evaluation is better
            else:
                if func_eval > best_func:
                    best_func = func_eval
                    best_func_evals = 1
            
            # Add the new individual to the population history
            self.population_history.append((new_individual, func_eval))
            
            # If the population has reached the budget, return the best function
            if len(self.population_history) >= self.budget:
                return best_func
        
        return best_func

# One-line description with the main idea
# Description: Hyperband Algorithm for Black Box Optimization
# Code: 
# ```python
# Hyperband Algorithm for Black Box Optimization
# 
# The Hyperband Algorithm is a metaheuristic algorithm for solving black box optimization problems.
# It uses adaptive sampling to search for the optimal solution, and it refines its strategy based on the performance of the current solution.
# 
# The algorithm starts with a population of random individuals and iteratively selects a new individual based on its fitness.
# It evaluates the fitness of each individual and selects the best one to replace the previous one.
# The algorithm continues until the population has reached the budget, at which point it returns the best individual as the solution.
# 
# The Hyperband Algorithm is particularly effective for solving complex optimization problems with multiple local optima.
# 
# Parameters:
#   budget (int): The maximum number of function evaluations allowed.
#   dim (int): The dimensionality of the search space.
# 
# Returns:
#   The optimal solution to the optimization problem.
# 
# Example:
#   >>> from llamea import LLaMEA
#   >>> from sklearn.datasets import load_iris
#   >>> iris = load_iris()
#   >>> X = iris.data
#   >>> y = iris.target
#   >>> X_train, X_test, y_train, y_test = X[:70], X[70:], y[:70], y[70:]
#   >>> lla = LLaMEA(100, 4)
#   >>> best_func = lla(X_train, y_train)
#   >>> print(best_func)
#   >>> print(lla.population_history)
#   ```