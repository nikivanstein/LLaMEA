# Description: Improved Black Box Optimization using Evolutionary Algorithm
# Code: 
# ```python
import random
import numpy as np
from scipy.optimize import differential_evolution

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.population_size = 100
        self.crossover_rate = 0.5
        self.mutation_rate = 0.1

    def __call__(self, func):
        # Evaluate the function with the given budget
        func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)])
        
        # Select the top-performing individuals
        top_individuals = np.argsort(func_evaluations)[-self.population_size:]
        
        # Create a new population by crossover and mutation
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(top_individuals, 2)
            child = (parent1 + parent2) / 2
            if random.random() < self.mutation_rate:
                child = random.uniform(self.search_space[0], self.search_space[1])
            new_population.append(child)
        
        # Replace the old population with the new one
        self.population = new_population
        
        # Evaluate the new population
        new_func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(new_population))])
        
        # Return the best individual
        best_individual = np.argmax(new_func_evaluations)
        return new_population[best_individual]

def improved_black_box_optimizer(budget, dim):
    # Initialize the evolutionary algorithm
    optimizer = BlackBoxOptimizer(budget, dim)
    
    # Run the evolutionary algorithm
    best_individual = optimizer()
    
    # Print the result
    print(f"Best individual: {best_individual}")
    print(f"Best fitness: {np.max([func(x) for x in best_individual])}")
    
    # Evaluate the function with the given budget
    func_evaluations = np.array([best_individual(x) for x in np.random.uniform(optimizer.search_space[0], optimizer.search_space[1], 100)])
    
    # Select the top-performing individuals
    top_individuals = np.argsort(func_evaluations)[-optimizer.population_size:]
    
    # Create a new population by crossover and mutation
    new_population = []
    for _ in range(optimizer.population_size):
        parent1, parent2 = random.sample(top_individuals, 2)
        child = (parent1 + parent2) / 2
        if random.random() < optimizer.mutation_rate:
            child = random.uniform(optimizer.search_space[0], optimizer.search_space[1])
        new_population.append(child)
    
    # Replace the old population with the new one
    optimizer.population = new_population
    
    # Evaluate the new population
    new_func_evaluations = np.array([best_individual(x) for x in np.random.uniform(optimizer.search_space[0], optimizer.search_space[1], len(new_population))])
    
    # Return the best individual
    best_individual = np.argmax(new_func_evaluations)
    return new_population[best_individual]

# Test the function
def test_function(x):
    return x[0]**2 + x[1]**2

best_individual = improved_black_box_optimizer(100, 2)
print(f"Best individual: {best_individual}")
print(f"Best fitness: {np.max([test_function(x) for x in best_individual])}")