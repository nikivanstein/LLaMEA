import numpy as np
from scipy.optimize import differential_evolution

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

    def select_strategy(self, func, budget):
        # Select a strategy based on the budget and dimension
        if budget <= 10:
            strategy = "random"
        elif budget <= 50:
            strategy = "adaptive"
        else:
            strategy = "expert"
        
        return strategy

    def run(self, func, budget, dim):
        strategy = self.select_strategy(func, budget)
        
        if strategy == "random":
            # Run the function with random sampling
            new_individual = np.random.uniform(-5.0, 5.0, size=self.dim)
        elif strategy == "adaptive":
            # Run the function with adaptive sampling
            new_individual = np.random.uniform(-5.0, 5.0, size=self.dim)
        else:
            # Run the function with expert strategy
            new_individual = func(np.random.uniform(-5.0, 5.0, size=self.dim))
        
        # Evaluate the function at the new individual
        func_eval = func(new_individual)
        
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
        np.save(f"{self.sample_dir}_{self.sample_size}_{self.sample_size}_{self.sample_size}_{_}", new_individual)
        
        return self.best_func

# One-line description with the main idea
Hyperband Algorithm: A metaheuristic algorithm that uses adaptive sampling to optimize black box functions, with a focus on handling a wide range of tasks and tasks with varying dimensionality.

# Code: