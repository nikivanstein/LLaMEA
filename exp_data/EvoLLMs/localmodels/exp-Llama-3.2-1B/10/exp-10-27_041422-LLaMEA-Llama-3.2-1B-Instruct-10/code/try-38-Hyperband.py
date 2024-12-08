import numpy as np
from scipy.stats import norm
from scipy.optimize import differential_evolution

class Hyperband:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_func = None
        self.best_func_evals = 0
        self.sample_size = 1
        self.sample_dir = None

    def __call__(self, func, initial_individual, logger):
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
            sample = initial_individual
            
            # Evaluate the function at the current sample
            func_eval = func(sample)
            
            # If this is the first evaluation, update the best function
            if self.best_func_evals == 1:
                self.best_func = func_eval
                self.best_func_evals = 1
            # Otherwise, update the best function if the current evaluation is better
            else:
                # Use Bayesian optimization to update the best function
                bounds = [(-5.0, 5.0), (-5.0, 5.0)]
                res = differential_evolution(lambda x: -x[0]**2 + x[1]**2, bounds, x0=initial_individual, maxiter=1000)
                if res.fun < self.best_func_evals:
                    self.best_func = res.x[0]*norm.rvs(loc=0, scale=1, size=1) + res.x[1]*norm.rvs(loc=0, scale=1, size=1)
                    self.best_func_evals = res.fun
                else:
                    self.best_func_evals += 1
            
            # Save the current sample to the sample directory
            np.save(f"{self.sample_dir}_{self.sample_size}_{_}", sample)
        
        return self.best_func

# Description: Adaptive hyperparameter tuning using adaptive sampling and Bayesian optimization
# Code: 
# ```python
# ```python
# 
# ```python
# 
# 