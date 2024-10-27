import numpy as np
import os

class Hyperband:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_func = None
        self.best_func_evals = 0
        self.sample_size = 1
        self.sample_dir = None
        self.sample_history = []

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
            self.sample_history.append((sample, func_eval))
        
        return self.best_func

class Individual:
    def __init__(self, dim, budget):
        self.dim = dim
        self.budget = budget
        self.f = None
        self.history = []

    def __call__(self, func):
        if self.f is not None:
            return self.f
        
        # Initialize the function and its history
        self.f = func
        self.history = []
        
        # Set the sample size and directory
        self.sample_size = 10
        self.sample_dir = f"sample_{self.sample_size}"
        
        # Perform adaptive sampling
        for _ in range(self.budget):
            # Generate a random sample of size self.sample_size
            sample = np.random.uniform(-5.0, 5.0, size=self.dim)
            
            # Evaluate the function at the current sample
            func_eval = func(sample)
            
            # Append the sample and function evaluation to the history
            self.history.append((sample, func_eval))
        
        return self.f

# Initialize the algorithm
algorithm = Hyperband(budget=1000, dim=10)

# Select the first individual
individual = Individual(dim=10, budget=1000)

# Evaluate the function of the selected individual
func_eval = individual(individual)
# Update the best function
algorithm(algorithm, func_eval)

# Print the best function and its score
print(f"Best function: {algorithm.best_func}")
print(f"Best function score: {algorithm.best_func_evals}")

# Refine the strategy of the selected individual
individual = Individual(dim=10, budget=1000)
func_eval = individual(individual)
# Update the best function
algorithm(algorithm, func_eval)

# Print the best function and its score
print(f"Best function: {algorithm.best_func}")
print(f"Best function score: {algorithm.best_func_evals}")

# Save the best function and its history to the sample directory
np.save(f"{algorithm.sample_dir}_best_func", algorithm.best_func)
np.save(f"{algorithm.sample_dir}_history", algorithm.history)