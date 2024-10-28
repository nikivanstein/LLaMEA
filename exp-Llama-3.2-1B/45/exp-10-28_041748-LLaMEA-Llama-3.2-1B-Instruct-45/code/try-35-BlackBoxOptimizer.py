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
        self.t_max = 1000

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

def hybrid_optimization(func, budget, dim, t=0):
    # Initialize the optimizer
    optimizer = BlackBoxOptimizer(budget, dim)
    
    # Initialize the best solution
    best_solution = None
    best_fitness = -np.inf
    
    # Initialize the temperature
    temperature = 1.0
    
    # Loop until the maximum number of iterations is reached
    for _ in range(t):
        # Evaluate the function with the given budget
        func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], optimizer.population_size)])
        
        # Select the top-performing individuals
        top_individuals = np.argsort(func_evaluations)[-optimizer.population_size:]
        
        # Create a new population by crossover and mutation
        new_population = []
        for _ in range(optimizer.population_size):
            parent1, parent2 = random.sample(top_individuals, 2)
            child = (parent1 + parent2) / 2
            if random.random() < optimizer.mutation_rate:
                child = random.uniform(self.search_space[0], self.search_space[1])
            new_population.append(child)
        
        # Replace the old population with the new one
        optimizer.population = new_population
        
        # Evaluate the new population
        new_func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(new_population))])
        
        # Update the best solution
        best_individual = np.argmax(new_func_evaluations)
        if new_func_evaluations[best_individual] > best_fitness:
            best_solution = new_population[best_individual]
            best_fitness = new_func_evaluations[best_individual]
        
        # Decrease the temperature
        temperature *= 0.95
        
        # Check if the temperature has reached the minimum allowed value
        if temperature < 0.01:
            break
    
    # Return the best solution
    return best_solution

# Example usage:
def func(x):
    return np.sin(x)

best_solution = hybrid_optimization(func, 1000, 10)
print("Best solution:", best_solution)
print("Best fitness:", np.max([func(x) for x in best_solution]))