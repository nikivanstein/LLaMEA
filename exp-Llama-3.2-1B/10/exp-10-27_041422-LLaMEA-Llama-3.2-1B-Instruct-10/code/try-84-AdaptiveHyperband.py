import numpy as np
import random

class AdaptiveHyperband:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_func = None
        self.best_func_evals = 0
        self.sample_size = 10
        self.sample_dir = None
        self.exploration_rate = 0.1

    def __call__(self, func, num_evals):
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
        for _ in range(num_evals):
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
        
        # Increase the exploration rate for the next iteration
        self.exploration_rate *= 0.9
        
        return self.best_func

# Example usage
if __name__ == "__main__":
    # Define the black box function
    def func(x):
        return x**2
    
    # Create an AdaptiveHyperband object with 1000 function evaluations
    adaptive_hyperband = AdaptiveHyperband(1000, 10)
    
    # Optimize the function using AdaptiveHyperband
    best_func = adaptive_hyperband(func, 1000)
    
    # Print the score
    print(f"Best function: {best_func}")
    print(f"Score: {best_func.eval()}")