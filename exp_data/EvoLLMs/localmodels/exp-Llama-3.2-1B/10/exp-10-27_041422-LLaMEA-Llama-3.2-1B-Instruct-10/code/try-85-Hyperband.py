# Description: Hyperband Algorithm
# Code: 
# ```python
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
        self.search_space = (-5.0, 5.0, -5.0, 5.0)

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
        self.search_space = (-5.0, 5.0, -5.0, 5.0)
        self.population_size = 100
        self.population_size_per_eval = 10
        self.population_dir = None

    def __call__(self, func):
        # Initialize the best function and its evaluation count
        self.best_func = func
        self.best_func_evals = 0
        
        # Set the population size and directory
        self.population_size = self.population_size_per_eval
        self.population_dir = f"population_{self.population_size}"
        
        # Perform adaptive sampling
        for i in range(self.budget):
            # Generate a random population of size self.population_size
            population = np.random.uniform(self.search_space[0], self.search_space[1], size=self.population_size)
            
            # Evaluate the function at the current population
            func_evals = 0
            for individual in population:
                func_eval = func(individual)
                func_evals += 1
            avg_func_eval = func_evals / self.population_size
            
            # If this is the first evaluation, update the best function
            if self.best_func_evals == 0:
                self.best_func = func_evals
                self.best_func_evals = 1
            # Otherwise, update the best function if the current evaluation is better
            else:
                if avg_func_eval > self.best_func_evals:
                    self.best_func = func_evals
                    self.best_func_evals = 1
                else:
                    self.best_func_evals += 1
            
            # Save the current population to the population directory
            np.save(f"{self.population_dir}_{i}_{_}", population)
        
        return self.best_func

class HyperbandEvolutionaryAlgorithm(Hyperband):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.population_size = 100
        self.population_size_per_eval = 10
        self.population_dir = None
        self.population = None
        self.population_evals = 0
        self.population_dir_eval = None

    def __call__(self, func):
        # Initialize the best function and its evaluation count
        self.best_func = func
        self.best_func_evals = 0
        
        # Set the population size and directory
        self.population_size = self.population_size_per_eval
        self.population_dir = f"population_{self.population_size}"
        
        # Perform adaptive sampling
        for i in range(self.budget):
            # Generate a random population of size self.population_size
            population = np.random.uniform(self.search_space[0], self.search_space[1], size=self.population_size)
            
            # Evaluate the function at the current population
            func_evals = 0
            for individual in population:
                func_eval = func(individual)
                func_evals += 1
            avg_func_eval = func_evals / self.population_size
            
            # If this is the first evaluation, update the best function
            if self.best_func_evals == 0:
                self.best_func = func_evals
                self.best_func_evals = 1
            # Otherwise, update the best function if the current evaluation is better
            else:
                if avg_func_eval > self.best_func_evals:
                    self.best_func = func_evals
                    self.best_func_evals = 1
                else:
                    self.best_func_evals += 1
            
            # Save the current population to the population directory
            np.save(f"{self.population_dir}_{i}_{_}", population)
        
        # Initialize the population
        self.population = population
        
        # Initialize the population evaluations
        self.population_evals = 0
        
        # Evaluate the population
        while self.population_evals < self.budget:
            # Evaluate the function at the current population
            func_evals = 0
            for individual in self.population:
                func_eval = func(individual)
                func_evals += 1
            avg_func_eval = func_evals / len(self.population)
            
            # If this is the first evaluation, update the best function
            if self.best_func_evals == 0:
                self.best_func = func_evals
                self.best_func_evals = 1
            # Otherwise, update the best function if the current evaluation is better
            else:
                if avg_func_eval > self.best_func_evals:
                    self.best_func = func_evals
                    self.best_func_evals = 1
                else:
                    self.best_func_evals += 1
            
            # Save the current population to the population directory
            np.save(f"{self.population_dir_eval}_{self.population_evals}_{_}", self.population)
            
            # Increment the population evaluations
            self.population_evals += 1
            
            # Limit the population to a maximum size
            if len(self.population) > self.population_size:
                self.population = self.population[:self.population_size]
        
        return self.best_func

# Example usage:
func = lambda x: np.sin(x)
hyperband = HyperbandHyperband(100, 10)
best_func = hyperband(func)
print(best_func)