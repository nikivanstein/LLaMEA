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
        self.best_individual = None

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
        
        # Refine the strategy
        self.refine_strategy()
        
        return self.best_func

    def refine_strategy(self):
        # Define the strategy for the current best individual
        strategy = {
            "type": "random",
            "params": {
                "population_size": self.sample_size,
                "crossover_rate": 0.8,
                "mutation_rate": 0.1
            }
        }
        
        # Initialize the new population
        new_population = self.evaluate_fitness(strategy)
        
        # Replace the current best individual with the new population
        self.best_individual = new_population[0]
        
        # Update the best function and its evaluation count
        self.best_func = self.evaluate_fitness(self.best_individual)
        self.best_func_evals = 1
        
        # Save the new population to the sample directory
        np.save(f"{self.sample_dir}_{self.sample_size}_{self.best_individual}", self.best_individual)

    def evaluate_fitness(self, func):
        # Evaluate the function at a given individual
        return func(self.best_individual)

# Description: Hyperband algorithm
# Code: 