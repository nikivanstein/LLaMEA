# Description: Novel Metaheuristic Algorithm for Black Box Optimization Using Genetic Programming and Evolution Strategies
# Code:
import random
import numpy as np
import math

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.num_samples = 0
        self.func_values = {}
        self.population = []
        self.fitness_values = {}

    def __call__(self, func):
        # Initialize the population with random solutions
        for _ in range(100):
            individual = [random.uniform(self.search_space[0], self.search_space[1]) for _ in range(self.dim)]
            self.population.append(individual)
        
        # Evaluate the function 1 time
        for individual in self.population:
            func(individual)
        
        # Select the fittest individuals
        self.fittest_individuals = sorted(self.population, key=lambda x: self.fitness_values[x], reverse=True)[:self.budget]
        
        # Create new individuals by mutation and crossover
        for _ in range(self.budget):
            parent1 = random.choice(self.fittest_individuals)
            parent2 = random.choice(self.fittest_individuals)
            child = [x + y for x, y in zip(parent1, parent2)]
            if random.random() < 0.5:
                child.insert(0, parent1[0])
            self.population.append(child)
        
        # Evaluate the function 1 time
        for individual in self.population:
            func(individual)
        
        # Update the fitness values
        self.fitness_values = {}
        for individual in self.population:
            self.fitness_values[individual] = func(individual)

    def __str__(self):
        return f"Genetic Algorithm with fittest individuals: {self.fittest_individuals}\nFitness values: {self.fitness_values}"

# Description: A simple genetic algorithm for black box optimization
# Code: 
# Algorithm: Genetic Algorithm
# Description: A simple genetic algorithm for black box optimization using crossover and mutation
# Code:
# The genetic algorithm is initialized with a population of random solutions, evaluates the function 1 time, 
# selects the fittest individuals, creates new individuals by crossover and mutation, and evaluates the function 1 time.
# The fitness values are updated after each evaluation.