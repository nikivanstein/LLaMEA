# Description: Hyperband Algorithm for Black Box Optimization
# Code: 
# ```python
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
        
        return self.best_func

def evaluate_fitness(individual, func, budget):
    """Evaluates a black box function at a given individual."""
    return func(individual)

def select_strategy(individual, func, budget):
    """Selects a strategy based on the individual's fitness."""
    # Randomly sample a subset of the population
    sample_size = random.randint(1, budget)
    sample = random.sample(individual, sample_size)
    
    # Evaluate the function at the sampled individuals
    fitness_values = [evaluate_fitness(individual, func, budget) for individual in sample]
    
    # Select the individual with the highest fitness
    return sample[np.argmax(fitness_values)]

def run_hyperband(individual, func, budget, population_size):
    """Runs the Hyperband algorithm."""
    population = [individual] * population_size
    for _ in range(budget):
        individual = select_strategy(population, func, budget)
        population.append(individual)
    
    return population

# Test the Hyperband algorithm
def test_hyperband():
    func = lambda x: x**2
    budget = 10
    population_size = 100
    population = run_hyperband(random.randint(0, 100), func, budget, population_size)
    
    # Evaluate the best function in the population
    best_func = min(population, key=evaluate_fitness)
    best_func_evals = sum(1 for func_eval in evaluate_fitness(population, best_func, budget) if func_eval == best_func_evals)
    print(f"Best function: {best_func_evals} evaluations")
    print(f"Best function: {best_func} ({best_func_evals} evaluations)")

test_hyperband()