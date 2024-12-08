# Description: Evolutionary Optimization for BBOB
# Code: 
# ```python
import random
import numpy as np

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
        return best_individual

def mutation_exp(population, func, mutation_rate, upper_limit):
    # Select the top-performing individuals
    top_individuals = np.argsort(func(population))[-mutation_rate * population.size:]
    
    # Create a new population by crossover and mutation
    new_population = []
    for _ in range(population.size):
        parent1, parent2 = random.sample(top_individuals, 2)
        child = (parent1 + parent2) / 2
        if random.random() < mutation_rate:
            child = random.uniform(upper_limit[0], upper_limit[1])
        new_population.append(child)
    
    # Replace the old population with the new one
    population = new_population

def selection_exp(population, func, budget, upper_limit):
    # Evaluate the function with the given budget
    func_evaluations = np.array([func(x) for x in np.random.uniform(upper_limit[0], upper_limit[1], budget)])
    
    # Select the top-performing individuals
    top_individuals = np.argsort(func_evaluations)[-budget:]
    
    # Create a new population by crossover and mutation
    new_population = []
    for _ in range(budget):
        parent1, parent2 = random.sample(top_individuals, 2)
        child = (parent1 + parent2) / 2
        if random.random() < 0.5:
            child = random.uniform(upper_limit[0], upper_limit[1])
        new_population.append(child)
    
    # Replace the old population with the new one
    population = new_population

# BBOB test suite
def bbooB():
    # Define the functions to optimize
    functions = [
        lambda x: np.sin(x),
        lambda x: np.cos(x),
        lambda x: np.tan(x),
        lambda x: np.exp(x),
        lambda x: np.log(x),
        lambda x: np.sqrt(x),
        lambda x: 1 / x,
        lambda x: np.cos(x / 2),
        lambda x: np.sin(x / 2),
        lambda x: np.tan(x / 2),
        lambda x: np.exp(x / 2),
        lambda x: np.log(x / 2),
        lambda x: np.sqrt(x / 2),
        lambda x: 1 / x / 2,
        lambda x: np.cos(x / 2 / 2),
        lambda x: np.sin(x / 2 / 2),
        lambda x: np.tan(x / 2 / 2),
        lambda x: np.exp(x / 2 / 2),
        lambda x: np.log(x / 2 / 2),
        lambda x: np.sqrt(x / 2 / 2),
        lambda x: 1 / x / 2 / 2,
    ]
    
    # Initialize the optimizer
    optimizer = BlackBoxOptimizer(100, 5)
    
    # Run the optimization algorithm
    best_individual = optimizer.__call__(bbooB)
    print("Best individual:", best_individual)
    
    # Print the fitness of the best individual
    print("Fitness:", np.mean([np.sin(x) for x in np.random.uniform(upper_limit[0], upper_limit[1], 100)]))