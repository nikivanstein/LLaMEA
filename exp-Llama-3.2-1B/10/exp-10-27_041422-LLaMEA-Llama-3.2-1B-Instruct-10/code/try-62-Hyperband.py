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
        self.sample_history = []

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
        
        return self.best_func

# Exception Handling
try:
    new_individual = Hyperband(100, 10).evaluate_fitness(np.random.uniform(-5.0, 5.0, size=10))
except Exception as e:
    print(f"An error occurred: {e}")
    new_individual = None

# Hyperband Algorithm Implementation
def hyperband_search(func, budget, dim, population_size=100, mutation_rate=0.01, max_iter=1000):
    # Initialize the population
    population = [np.random.uniform(-5.0, 5.0, size=dim) for _ in range(population_size)]
    
    # Initialize the best individual and its fitness
    best_individual = None
    best_fitness = float('-inf')
    
    # Perform the hyperband search
    for _ in range(max_iter):
        # Evaluate the fitness of each individual
        fitnesses = [func(individual) for individual in population]
        
        # Select the best individual
        selected_individual = population[np.argmax(fitnesses)]
        
        # Select a new individual using hyperband
        new_individual = None
        for _ in range(population_size):
            if random.random() < mutation_rate:
                new_individual = selected_individual + np.random.uniform(-5.0, 5.0, size=dim)
        
        # Update the best individual and its fitness
        if new_individual is not None and func(new_individual) > best_fitness:
            best_individual = new_individual
            best_fitness = func(new_individual)
        
        # Save the new individual to the sample history
        np.save(f"sample_{population_size}_{_}", new_individual)
        
        # Update the population
        population = [new_individual for _ in range(population_size)]
    
    return best_individual, best_fitness

# Example usage
def func1(x):
    return np.sum(x**2)

best_individual, best_fitness = hyperband_search(func1, 100, 10)

# Print the best individual and its fitness
print(f"Best individual: {best_individual}")
print(f"Best fitness: {best_fitness}")