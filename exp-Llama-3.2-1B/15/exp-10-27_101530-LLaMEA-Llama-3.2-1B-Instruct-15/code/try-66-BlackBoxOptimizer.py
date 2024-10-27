import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0

    def __call__(self, func):
        # Initialize population with random points in the search space
        population = self.generate_population(self.budget)
        
        # Evaluate the fitness of each individual in the population
        fitnesses = [func(individual) for individual in population]
        
        # Select the fittest individuals to reproduce
        fittest_individuals = self.select_fittest(population, fitnesses)
        
        # Create a new population by crossover and mutation
        new_population = self.crossover_and_mutate(fittest_individuals)
        
        # Evaluate the fitness of the new population
        new_fitnesses = [func(individual) for individual in new_population]
        
        # Replace the old population with the new one
        population = new_population
        
        # Update the population size and budget
        self.budget = max(1, self.budget - len(population))
        
        # Return the fittest individual in the new population
        return self.select_fittest(population, new_fitnesses)[-1]
        
    def generate_population(self, budget):
        # Generate 'budget' number of random points in the search space
        return [np.random.uniform(self.search_space[0], self.search_space[1]) for _ in range(budget)]
    
    def select_fittest(self, population, fitnesses):
        # Select the fittest individuals based on their fitness
        return sorted(population, key=lambda individual: fitnesses[fitnesses.index(individual)], reverse=True)[:self.budget]
    
    def crossover_and_mutate(self, fittest_individuals):
        # Perform crossover and mutation on the fittest individuals
        children = []
        while len(children) < len(fittest_individuals):
            parent1, parent2 = random.sample(fittest_individuals, 2)
            child = (parent1 + parent2) / 2
            if random.random() < 0.5:
                # Perform mutation on the child
                child = random.uniform(self.search_space[0], self.search_space[1]) + (np.random.uniform(-1, 1) * (child - self.search_space[0])) / 2
            children.append(child)
        return children

# One-line description with main idea
# Novel Metaheuristic Algorithm for Black Box Optimization using Evolutionary Strategies
# The algorithm uses a population of individuals with crossover and mutation, and selects the fittest individuals to reproduce and replace the old population.
# The fitness of each individual is evaluated, and the new population is updated with the fittest individuals.
# The process is repeated until the budget is reached.