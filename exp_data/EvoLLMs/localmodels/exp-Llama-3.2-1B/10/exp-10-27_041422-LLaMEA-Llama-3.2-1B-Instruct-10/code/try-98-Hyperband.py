import numpy as np
import random
from scipy.optimize import differential_evolution

class Hyperband:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_func = None
        self.best_func_evals = 0
        self.sample_size = 1
        self.sample_dir = None
        self.best_individual = None
        self.best_fitness = float('inf')

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
                if func_eval < self.best_func:
                    self.best_func = func_eval
                    self.best_func_evals = 1
                else:
                    self.best_func_evals += 1
            
            # Save the current sample to the sample directory
            np.save(f"{self.sample_dir}_{self.sample_size}_{_}", sample)
        
        # Refine the strategy using differential evolution
        self.refine_strategy()
        
        return self.best_func

    def refine_strategy(self):
        # Define the bounds for the differential evolution
        bounds = [(-5.0, 5.0) for _ in range(self.dim)]
        
        # Perform differential evolution
        result = differential_evolution(lambda x: -x, bounds, popcount=1000)
        
        # Update the best individual and fitness
        self.best_individual = result.x
        self.best_fitness = -result.fun
        
        # Update the best function and its evaluation count
        self.best_func = self.best_individual
        self.best_func_evals = 1
        
        # Update the sample directory
        np.save(f"{self.sample_dir}_{self.sample_size}_best_{self.best_individual}", self.best_individual)
        
        # Update the budget
        self.budget -= 1
        
        # If the budget is exhausted, reset the best function and strategy
        if self.budget <= 0:
            self.best_func = None
            self.best_func_evals = 0
            self.best_individual = None
            self.best_fitness = float('inf')