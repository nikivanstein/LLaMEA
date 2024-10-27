import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.population_size = 100
        self.population = self.generate_population()

    def generate_population(self):
        # Initialize the population with random individuals
        population = [random.uniform(self.search_space[0], self.search_space[1]) for _ in range(self.population_size)]
        return population

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Evaluate the fitness of each individual in the population
            fitnesses = [self.evaluate_fitness(individual, func) for individual in self.population]
            # Select the fittest individuals
            fittest_individuals = self.select_fittest(population, fitnesses)
            # Create a new population by crossover and mutation
            new_population = self.crossover_and_mutation(fittest_individuals, func)
            # Replace the old population with the new one
            self.population = new_population
            # Update the function evaluations
            self.func_evaluations += 1
        # Return the fittest individual found so far
        return self.population[0]

    def evaluate_fitness(self, individual, func):
        # Evaluate the function at the individual
        func_value = func(individual)
        return func_value

    def select_fittest(self, population, fitnesses):
        # Select the fittest individuals based on their fitness
        fittest_individuals = sorted(population, key=lambda individual: fitnesses[individual], reverse=True)
        return fittest_individuals[:self.population_size//2]

    def crossover_and_mutation(self, fittest_individuals, func):
        # Create a new population by crossover and mutation
        new_population = []
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(fittest_individuals, 2)
            child = (parent1 + parent2) / 2
            # Apply mutation to the child
            if random.random() < 0.2:
                child = self.mutate(child)
            new_population.append(child)
        return new_population

    def mutate(self, individual):
        # Apply mutation to the individual
        if random.random() < 0.1:
            return individual + random.uniform(-1, 1)
        return individual

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization using Genetic Programming
# to handle a wide range of tasks on the BBOB test suite of 24 noiseless functions.