import numpy as np
from scipy.optimize import differential_evolution
import os
import pickle

class Hyperband:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_func = None
        self.best_func_evals = 0
        self.sample_size = 1
        self.sample_dir = None

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
            np.save(f"{self.sample_dir}_{self.sample_size}_{_}", sample)
        
        return self.best_func

# One-line description with main idea
# Hyperband: Adaptive Black Box Optimization using Hyperparameter Tuning
# Code: 
# ```python
def optimize_bbob(func, budget, dim):
    """Optimize a black box function using Hyperband algorithm."""
    hyperband = Hyperband(budget, dim)
    return hyperband(func)

# Test the function
def test_func1(x):
    return x[0]**2 + x[1]**2

def test_func2(x):
    return x[0]**3 + x[1]**3

def test_func3(x):
    return x[0]*x[1]

# Evaluate the functions
func1 = optimize_bbob(test_func1, 100, 2)
func2 = optimize_bbob(test_func2, 100, 2)
func3 = optimize_bbob(test_func3, 100, 2)

# Print the results
print("Optimized Function 1:", func1)
print("Optimized Function 2:", func2)
print("Optimized Function 3:", func3)

# Save the results
with open("results.pkl", "wb") as f:
    pickle.dump((func1, func2, func3), f)