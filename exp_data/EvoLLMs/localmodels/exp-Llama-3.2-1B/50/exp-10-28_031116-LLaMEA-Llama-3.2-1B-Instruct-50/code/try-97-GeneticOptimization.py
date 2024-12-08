import numpy as np
import random

class GeneticOptimization:
    def __init__(self, budget, dim, mutation_rate=0.01, crossover_rate=0.8):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population_size = 100
        self.population = self.initialize_population()

    def initialize_population(self):
        return [[random.uniform(-5.0, 5.0) for _ in range(self.dim)] for _ in range(self.population_size)]

    def __call__(self, func):
        # Evaluate the function
        func_value = func(self.population)
        # Select the fittest individuals
        fittest_individuals = self.select_fittest_individuals(func_value, self.population_size, self.budget)
        # Create a new generation
        new_population = self.create_new_generation(fittest_individuals, self.dim, self.budget)
        # Calculate the fitness of each individual in the new generation
        fitness = [self.calculate_fitness(individual, func_value) for individual in new_population]
        # Select the fittest individuals for the next generation
        self.population = self.select_fittest_individuals(fitness, self.population_size, self.budget)
        # Return the fittest individual
        return self.population[0]

    def select_fittest_individuals(self, fitness, population_size, budget):
        # Select the fittest individuals using tournament selection
        tournament_size = 3
        winners = []
        for _ in range(population_size):
            winner = random.choices(fitness, weights=fitness, k=tournament_size)[0]
            winners.append(winner)
        return winners

    def create_new_generation(self, fittest_individuals, dim, budget):
        # Create a new generation by crossover and mutation
        new_generation = []
        for _ in range(budget):
            parent1 = random.choice(fittest_individuals)
            parent2 = random.choice(fittest_individuals)
            child = self.crossover(parent1, parent2, dim)
            child = self.mutate(child, self.mutation_rate)
            new_generation.append(child)
        return new_generation

    def crossover(self, parent1, parent2, dim):
        # Perform crossover between two parents
        child = [random.choice(parent1) for _ in range(dim)]
        for i in range(dim):
            if random.random() < self.crossover_rate:
                child[i] = random.choice(parent2)
        return child

    def mutate(self, individual, mutation_rate):
        # Perform mutation on an individual
        mutated_individual = [random.choice([x for x in individual if x!= random.choice(x)]) for _ in range(len(individual))]
        if random.random() < mutation_rate:
            mutated_individual[random.randint(0, len(individual) - 1)] = random.choice([x for x in individual if x!= random.choice(x)])
        return mutated_individual

    def calculate_fitness(self, individual, func_value):
        # Calculate the fitness of an individual
        return func_value(individual)

    def fitness_score(self, func_value):
        # Calculate the fitness score of a function
        return func_value

# Description: Genetic Optimization Algorithm using Genetic Programming
# Code: 