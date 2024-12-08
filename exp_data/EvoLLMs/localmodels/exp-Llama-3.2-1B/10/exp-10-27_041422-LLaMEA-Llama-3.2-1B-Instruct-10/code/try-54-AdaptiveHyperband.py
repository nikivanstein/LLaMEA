import numpy as np
import random

class AdaptiveHyperband:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_func = None
        self.best_func_evals = 0
        self.sample_size = 10
        self.sample_dir = f"sample_{self.sample_size}"
        self.sample_history = []
        self.sample_dir_history = []

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
            self.sample_dir_history.append(f"sample_{self.sample_size}_{_}")
            
            # Save the current sample to the sample directory
            np.save(f"{self.sample_dir}_{self.sample_size}_{_}", sample)
        
        return self.best_func

    def update(self, new_individual):
        # Refine the strategy by changing the sample size and sampling distribution
        new_sample_size = random.randint(5, 20)
        new_sample_dir = f"sample_{new_sample_size}"
        
        # Update the sample history and directory
        self.sample_history = [sample for sample in self.sample_history if sample.shape[0] < new_sample_size]
        self.sample_dir_history = [directory for directory in self.sample_dir_history if directory.startswith(f"sample_{new_sample_size}_{_}")]
        
        # Update the best function
        new_individual = self.evaluate_fitness(new_individual)
        
        # Save the updated sample to the sample directory
        np.save(f"{new_sample_dir}_{new_sample_size}_{_}", new_individual)
        
        return new_individual

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of the individual
        func_eval = self.f(individual, self.logger)
        
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
        
        return func_eval