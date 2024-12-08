import numpy as np
import random

class AdaptiveHyperband:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_func = None
        self.best_func_evals = 0
        self.sample_size = 1
        self.sample_dir = None
        self.sample_history = []
        self.sample_history_best_func = None
        self.sample_history_best_func_evals = 0

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
            self.sample_history_best_func = func_eval
            
            # Check if we have enough samples to converge
            if len(self.sample_history) >= self.sample_size:
                # Calculate the average of the last sample
                avg_sample = np.mean(self.sample_history[-self.sample_size:])
                
                # Update the best function if the average sample is better
                if func_eval > avg_sample:
                    self.best_func = func_eval
                    self.best_func_evals = 1
                else:
                    self.best_func_evals += 1

        return self.best_func

# One-line description with the main idea
# AdaptiveHyperband: An adaptive hyperband optimization algorithm that uses adaptive sampling to refine its strategy.