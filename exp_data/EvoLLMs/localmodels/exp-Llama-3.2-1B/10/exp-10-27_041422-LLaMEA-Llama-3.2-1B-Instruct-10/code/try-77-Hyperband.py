# Description: Hyperband Algorithm
# Code: 
# ```python
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
        self.exploration_rate = 0.1  # Exploration rate for refinement

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
        
        # Refine the strategy using exploration
        for _ in range(int(self.budget * self.exploration_rate)):
            # Generate a new sample using the current best function
            new_sample = np.random.uniform(-5.0, 5.0, size=self.dim)
            
            # Evaluate the new function
            new_func_eval = func(new_sample)
            
            # If the new function is better, update the best function
            if new_func_eval > self.best_func:
                self.best_func = new_func_eval
                self.best_func_evals = 1
            
            # Save the new sample to the sample directory
            np.save(f"{self.sample_dir}_{self.sample_size}_{_}", new_sample)
        
        return self.best_func

# One-line description with main idea
# Hyperband Algorithm