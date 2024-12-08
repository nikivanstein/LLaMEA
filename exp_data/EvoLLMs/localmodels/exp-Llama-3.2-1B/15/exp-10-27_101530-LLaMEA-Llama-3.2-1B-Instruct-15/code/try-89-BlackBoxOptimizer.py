import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.population_size = 100
        self.mutation_rate = 0.01

    def __call__(self, func):
        # Initialize the population with random points in the search space
        population = [self.generate_point(self.search_space) for _ in range(self.population_size)]
        
        # Evaluate the fitness of each individual
        fitnesses = [self.evaluate_fitness(individual, func) for individual in population]
        
        # Select the fittest individuals
        fittest_individuals = self.select_fittest(population, fitnesses)
        
        # Create a new generation by crossover and mutation
        new_population = self.crossover_and_mutation(fittest_individuals, population, fitnesses)
        
        # Evaluate the new population
        new_fitnesses = [self.evaluate_fitness(individual, func) for individual in new_population]
        
        # Select the fittest individuals from the new population
        new_fittest_individuals = self.select_fittest(new_population, new_fitnesses)
        
        # Replace the old population with the new one
        population = new_population
        
        # Update the best individual
        self.population_size = min(self.population_size, len(population))
        best_individual = max(population, key=self.evaluate_fitness)
        
        # Update the budget
        self.budget -= len(population)
        
        # If the budget is reached, return the best individual found so far
        if self.budget <= 0:
            return best_individual
        
        return best_individual

    def generate_point(self, search_space):
        return (random.uniform(search_space[0], search_space[1]), random.uniform(search_space[0], search_space[1]))

    def evaluate_fitness(self, individual, func):
        func_value = func(individual)
        return func_value

    def select_fittest(self, population, fitnesses):
        # Use tournament selection to select the fittest individuals
        fittest_individuals = []
        for _ in range(len(population)):
            tournament_size = random.randint(1, len(population))
            winners = random.choices(population, weights=fitnesses, k=tournament_size)
            winner = max(winners, key=self.evaluate_fitness)
            fittest_individuals.append(winner)
        return fittest_individuals

    def crossover_and_mutation(self, parents, population, fitnesses):
        # Use crossover and mutation to create new individuals
        new_population = []
        for _ in range(len(population)):
            parent1, parent2 = random.sample(parents, 2)
            child = self.crossover(parent1, parent2)
            child_fitness = self.evaluate_fitness(child, fitnesses)
            new_population.append((child, child_fitness))
        return new_population

    def crossover(self, parent1, parent2):
        # Use uniform crossover to create a new individual
        crossover_point = random.randint(1, len(parent1) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

    def mutate(self, individual):
        # Use bit-flipping mutation to create a new individual
        bit_index = random.randint(0, len(individual) - 1)
        individual[bit_index] ^= 1
        return individual