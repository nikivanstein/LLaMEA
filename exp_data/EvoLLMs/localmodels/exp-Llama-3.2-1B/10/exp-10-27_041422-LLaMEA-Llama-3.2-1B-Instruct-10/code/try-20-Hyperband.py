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
        self.sample_count = 0

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
            
            # Update the sample count
            self.sample_count += 1
            
            # Save the current sample to the sample directory
            np.save(f"{self.sample_dir}_{self.sample_size}_{self.sample_count}", sample)
        
        return self.best_func

class AdaptiveHyperband(Hyperband):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.sample_dir = f"adaptive_hyperband_{self.sample_size}"
        self.sample_count = 0

    def __call__(self, func):
        # Refine the strategy by changing the sample size and directory based on the number of evaluations
        if self.sample_count < 10:
            self.sample_size = 5
            self.sample_dir = f"adaptive_hyperband_{self.sample_size}"
            self.sample_count = 0
        elif self.sample_count < 20:
            self.sample_size = 10
            self.sample_dir = f"adaptive_hyperband_{self.sample_size}"
            self.sample_count = 0
        else:
            self.sample_size = 20
            self.sample_dir = f"adaptive_hyperband_{self.sample_size}"
        
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
            
            # Update the sample count
            self.sample_count += 1
            
            # Save the current sample to the sample directory
            np.save(f"{self.sample_dir}_{self.sample_size}_{self.sample_count}", sample)
        
        return self.best_func

# Example usage
from sklearn.datasets import make_boston
from sklearn.model_selection import train_test_split

# Generate sample data
boston_data = make_boston()
X, y = boston_data.data, boston_data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the AdaptiveHyperband class
adaptive_hyperband = AdaptiveHyperband(10, 5)

# Evaluate the best function for each individual
best_individuals = []
for _ in range(10):
    func = np.random.uniform(-10, 10, size=5)
    best_individual = adaptive_hyperband(func)
    best_individuals.append(best_individual)

# Print the best individual for each function
for i, individual in enumerate(best_individuals):
    print(f"Function {i+1}: {individual}")