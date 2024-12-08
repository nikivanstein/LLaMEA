import numpy as np
import random

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
            np.save(f"{self.sample_dir}_{self.sample_size}_{_}", sample)
        
        # Refine the strategy based on the sample history
        self.refine_strategy()
        
        return self.best_func

    def refine_strategy(self):
        # Calculate the proportion of new evaluations that are better than the best function
        proportion = self.sample_history[1] / (self.sample_history[0] + self.sample_history[1])
        
        # If the proportion is less than 0.1, change the strategy to adaptive sampling
        if proportion < 0.1:
            self.sample_size = 5
            self.sample_dir = f"sample_{self.sample_size}"
            self.sample_history = []
        
        # Otherwise, keep the current sample size and directory
        else:
            self.sample_size = self.sample_size
        
        # Save the updated sample history
        self.sample_history.append(self.sample_size)

# One-line description with the main idea
# Hyperband: Adaptive Population-Based Optimization with Adaptive Sampling
# A metaheuristic algorithm that uses adaptive sampling and refinement based on the sample history to optimize black box functions.