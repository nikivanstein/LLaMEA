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
        self.search_space = (-5.0, 5.0)

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
            sample = np.random.uniform(self.search_space[0], self.search_space[1], size=self.dim)
            
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

class HyperbandHyperband(Hyperband):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.search_space = (-5.0, 5.0)
        self.sample_size = 10
        self.sample_dir = f"sample_{self.sample_size}"

    def __call__(self, func):
        # Refine the strategy
        if random.random() < 0.1:
            # Change the individual lines of the selected solution
            self.best_func = func(np.random.uniform(self.search_space[0], self.search_space[1], size=self.dim))
        
        return super().__call__(func)

class HyperbandEvolved(Hyperband):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.sample_size = 10
        self.sample_dir = f"sample_{self.sample_size}"

    def __call__(self, func):
        # Evolve the population
        for _ in range(100):
            # Select a parent using tournament selection
            parent = np.random.choice(self.budget, size=self.sample_size, replace=False)
            
            # Perform crossover
            child = np.random.uniform(self.search_space[0], self.search_space[1], size=self.dim)
            for i in range(self.sample_size):
                if random.random() < 0.5:
                    child[i] = parent[i]
            
            # Evaluate the child
            func_eval = func(child)
            
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
        
        return self.best_func

# Description: Hyperband Algorithm
# Code: 