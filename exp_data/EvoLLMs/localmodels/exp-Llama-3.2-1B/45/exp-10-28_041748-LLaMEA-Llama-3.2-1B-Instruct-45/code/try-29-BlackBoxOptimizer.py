import numpy as np
import random

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
        
        # Refine the strategy based on the updated fitness
        refined_individuals = []
        for individual in new_population:
            l2 = np.mean([func(x, self.logger) for x in np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)])
            if random.random() < 0.45:
                refined_individuals.append(individual)
            else:
                refined_individuals.append(refined_individuals[-1] if refined_individuals else individual)
        
        # Replace the old population with the refined one
        self.population = refined_individuals
        
        # Evaluate the refined population
        refined_func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(refined_individuals))])
        
        # Return the best individual
        best_individual = np.argmax(refined_func_evaluations)
        return refined_individuals[best_individual]

class Mutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.mutation_rate = 0.1

    def __call__(self, individual):
        # Select a random individual
        individual_lines = individual
        
        # Randomly select two lines
        line1, line2 = random.sample(individual_lines, 2)
        
        # Create a new individual by crossover and mutation
        new_individual = (line1 + line2) / 2
        
        # Randomly mutate the new individual
        if random.random() < self.mutation_rate:
            new_individual = random.uniform(self.search_space[0], self.search_space[1])
        
        return new_individual

class Selection:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.population_size = 100
        self.crossover_rate = 0.5

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
        
        # Refine the strategy based on the updated fitness
        refined_individuals = []
        for individual in new_population:
            l2 = np.mean([func(x, self.logger) for x in np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)])
            if random.random() < 0.45:
                refined_individuals.append(individual)
            else:
                refined_individuals.append(refined_individuals[-1] if refined_individuals else individual)
        
        # Replace the old population with the refined one
        self.population = refined_individuals
        
        # Evaluate the refined population
        refined_func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(refined_individuals))])
        
        # Return the best individual
        best_individual = np.argmax(refined_func_evaluations)
        return refined_individuals[best_individual]

# Initialize the algorithm
optimizer = BlackBoxOptimizer(1000, 10)

# Initialize the mutation and selection algorithms
mutation = Mutation(1000, 10)
selection = Selection(1000, 10)

# Run the algorithm
best_individual = None
best_fitness = -np.inf
for _ in range(100):
    # Select a random individual
    individual = selection.__call__(optimizer)
    
    # Evaluate the individual
    func = lambda x: optimizer.func(x)
    fitness = func(individual)
    
    # Refine the strategy based on the updated fitness
    refined_individuals = [individual]
    for _ in range(10):
        refined_individuals.append(Mutation(1000, 10).__call__(individual))
    
    # Replace the old population with the refined one
    individual = refined_individuals[-1]
    
    # Update the best individual and fitness
    if fitness > best_fitness:
        best_individual = individual
        best_fitness = fitness
    elif fitness == best_fitness:
        best_individual = refined_individuals[-1]
        best_fitness = fitness

# Print the result
print("Best Individual:", best_individual)
print("Best Fitness:", best_fitness)