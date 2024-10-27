# Description: Adaptive Multi-Step Hypervolume (AMSH) Optimization
# Code: 
# ```python
import numpy as np
import os
from scipy.optimize import minimize

class AMSH:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_func = None
        self.best_func_evals = 0
        self.sample_size = 10
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
            self.sample_history.append(sample)
        
        return self.best_func

    def select_strategy(self):
        # Refine the strategy using a probability of 0.1
        strategy = np.random.choice(['uniform', 'adaptive'], p=[0.4, 0.6])
        
        # Update the sample history based on the selected strategy
        if strategy == 'uniform':
            for _ in range(self.budget):
                sample = np.random.uniform(-5.0, 5.0, size=self.dim)
                self.sample_history.append(sample)
        elif strategy == 'adaptive':
            for _ in range(self.budget):
                sample = np.random.uniform(-5.0, 5.0, size=self.dim)
                func_eval = func(sample)
                if self.best_func_evals == 1:
                    self.best_func = func_eval
                    self.best_func_evals = 1
                else:
                    if func_eval > self.best_func:
                        self.best_func = func_eval
                        self.best_func_evals = 1
                    else:
                        self.best_func_evals += 1
            
            # Save the current sample to the sample directory
            self.sample_history.append(sample)
        
        # Update the best function and its evaluation count
        self.best_func = None
        self.best_func_evals = 0
        
        # Save the updated sample history to the sample directory
        np.save(f"{self.sample_dir}_history", self.sample_history)

# Description: Black Box Optimization using Adaptive Multi-Step Hypervolume (AMSH)
# Code: 
# ```python
ams = AMSH(budget=100, dim=5)
ams.select_strategy()