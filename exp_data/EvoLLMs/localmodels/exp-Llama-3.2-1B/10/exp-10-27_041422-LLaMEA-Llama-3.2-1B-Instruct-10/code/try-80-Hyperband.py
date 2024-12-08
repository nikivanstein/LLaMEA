# Description: Hyperband Algorithm
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
            if os.path.exists(self.sample_dir):
                os.remove(os.path.join(self.sample_dir, f"sample_{self.sample_size}_{_}.npy"))
            np.save(os.path.join(self.sample_dir, f"sample_{self.sample_size}_{_}.npy"), sample)
        
        return self.best_func

# Description: Hyperband Algorithm
# Code: 
# ```python
def hyperband(budget, dim):
    best_func = None
    best_func_evals = 0
    sample_size = 10
    sample_dir = f"sample_{sample_size}"
    
    for _ in range(budget):
        # Generate a random sample of size sample_size
        sample = np.random.uniform(-5.0, 5.0, size=dim)
        
        # Evaluate the function at the current sample
        func_eval = np.random.uniform(-5.0, 5.0, size=dim)
        
        # If this is the first evaluation, update the best function
        if best_func_evals == 0:
            best_func = func_eval
            best_func_evals = 1
        
        # Otherwise, update the best function if the current evaluation is better
        elif func_eval > best_func:
            best_func = func_eval
            best_func_evals = 1
        
        # Save the current sample to the sample directory
        if os.path.exists(sample_dir):
            os.remove(os.path.join(sample_dir, f"sample_{sample_size}_{_}.npy"))
        np.save(os.path.join(sample_dir, f"sample_{sample_size}_{_}.npy"), sample)
    
    return best_func

# Description: Hyperband Algorithm
# Code: 
# ```python
def hyperband_revised(budget, dim):
    best_func = None
    best_func_evals = 0
    sample_size = 10
    sample_dir = f"sample_{sample_size}"
    
    for _ in range(budget):
        # Generate a random sample of size sample_size
        sample = np.random.uniform(-5.0, 5.0, size=dim)
        
        # Evaluate the function at the current sample
        func_eval = np.random.uniform(-5.0, 5.0, size=dim)
        
        # If this is the first evaluation, update the best function
        if best_func_evals == 0:
            best_func = func_eval
            best_func_evals = 1
        
        # Otherwise, update the best function if the current evaluation is better
        elif np.random.rand() < 0.1:  # 10% chance to refine the strategy
            refined_func_eval = func_eval
            refined_func_evals = 1
        else:
            refined_func_eval = best_func
            refined_func_evals = best_func_evals
        
        # Save the current sample to the sample directory
        if os.path.exists(sample_dir):
            os.remove(os.path.join(sample_dir, f"sample_{sample_size}_{_}.npy"))
        np.save(os.path.join(sample_dir, f"sample_{sample_size}_{_}.npy"), sample)
    
    return best_func

# Example usage:
budget = 1000
dim = 10
best_func = hyperband(budget, dim)
print(f"Best function: {best_func}")
print(f"Best function evaluations: {best_func_evals}")

best_func_revised = hyperband_revised(budget, dim)
print(f"Best function revised: {best_func_revised}")
print(f"Best function evaluations revised: {best_func_revised_evals}")