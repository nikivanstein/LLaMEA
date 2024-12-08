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
        self.sample_strategy = "uniform"

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
        
        # Initialize the sample strategy
        if random.random() < 0.1:
            self.sample_strategy = "uniform"
        elif random.random() < 0.2:
            self.sample_strategy = "grid"
        else:
            self.sample_strategy = "adaptive"
        
        # Perform adaptive sampling
        for _ in range(self.budget):
            # Generate a random sample of size self.sample_size
            if self.sample_strategy == "uniform":
                sample = np.random.uniform(-5.0, 5.0, size=self.dim)
            elif self.sample_strategy == "grid":
                sample = np.random.uniform(-5.0, 5.0, size=self.dim) / 2
            else:
                sample = np.random.uniform(-5.0, 5.0, size=self.dim) * 0.1
            
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

# Example usage:
if __name__ == "__main__":
    # Create a black box function
    def func(x):
        return np.sin(x)
    
    # Create an instance of the AdaptiveHyperband algorithm
    adaptive_hyperband = AdaptiveHyperband(100, 2)
    
    # Optimize the function using the adaptive hyperband algorithm
    best_func = adaptive_hyperband(func)
    print("Best function:", best_func)
    print("Best function evaluation:", best_func_evals)
    print("Budget:", adaptive_hyperband.budget)
    print("Sample strategy:", adaptive_hyperband.sample_strategy)
    
    # Save the best function to a file
    np.save(f"best_func_{adaptive_hyperband.sample_dir}_{adaptive_hyperband.sample_size}", best_func)