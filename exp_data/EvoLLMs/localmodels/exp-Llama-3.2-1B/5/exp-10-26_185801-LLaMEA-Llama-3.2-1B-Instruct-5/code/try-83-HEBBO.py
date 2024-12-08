import numpy as np
import random

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population_size = 100
        self.mutation_rate = 0.01

    def __call__(self, func):
        # Initialize population with random solutions
        population = [self.generate_random_solution(func, self.search_space) for _ in range(self.population_size)]
        
        # Evaluate population fitness
        population_fitness = [self.evaluate_fitness(population, func) for population in population]
        
        # Select fittest individuals
        fittest_individuals = sorted(zip(population_fitness, population), reverse=True)[:self.population_size // 2]
        
        # Create new population by mutation
        new_population = []
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(fittest_individuals, 2)
            child = (parent1 + parent2) / 2
            if random.random() < self.mutation_rate:
                child = random.uniform(self.search_space)
            new_population.append(child)
        
        # Replace old population with new one
        population = new_population
        
        # Update population with new fitness
        population_fitness = [self.evaluate_fitness(population, func) for population in population]
        
        # Select fittest individuals
        fittest_individuals = sorted(zip(population_fitness, population), reverse=True)[:self.population_size // 2]
        
        # Return fittest individual
        return fittest_individuals[0][1]

    def generate_random_solution(self, func, search_space):
        return np.random.uniform(search_space)

    def evaluate_fitness(self, population, func):
        return [func(individual) for individual in population]

# One-line description with main idea
# Evolutionary Algorithm for Black Box Optimization
# Using Genetic Algorithm to optimize black box functions