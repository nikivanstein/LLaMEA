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
            
            # Save the current sample to the sample directory
            np.save(f"{self.sample_dir}_{self.sample_size}_{_}", sample)
        
        return self.best_func

    def update_strategy(self, new_individual, func):
        # Refine the strategy by changing the individual lines
        for i in range(self.dim):
            if np.random.rand() < 0.1:
                new_individual[i] = np.random.uniform(-5.0, 5.0)
        
        # Evaluate the new individual using the budget
        new_func_eval = func(new_individual)
        
        # Update the best function if the new evaluation is better
        if new_func_eval > self.best_func:
            self.best_func = new_func_eval
            self.best_func_evals = 1
        else:
            self.best_func_evals += 1
        
        # Save the new individual to the sample directory
        np.save(f"{self.sample_dir}_{self.sample_size}_{_}", new_individual)
        
        # Update the sample history
        self.sample_history.append((new_individual, new_func_eval))
        
        return new_func_eval

# One-line description with the main idea
# Hyperband Algorithm: A metaheuristic that combines adaptive sampling and probability-based strategy refinement for efficient black box optimization.
# Code: 