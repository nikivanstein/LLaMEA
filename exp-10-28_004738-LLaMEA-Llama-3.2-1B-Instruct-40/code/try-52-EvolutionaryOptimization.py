import random
import numpy as np
from scipy.optimize import minimize

class EvolutionaryOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.funcs = self.generate_functions()

    def generate_functions(self):
        functions = []
        for _ in range(24):
            func = lambda x: random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def __call__(self, func, initial_individual, bounds, population_size, mutation_rate, selection_rate, crossover_rate, num_generations):
        # Initialize population with random individuals
        population = [initial_individual] * population_size
        for _ in range(num_generations):
            # Evaluate fitness of each individual
            fitnesses = [func(individual, self.funcs[individual]) for individual in population]
            # Select parents using tournament selection
            parents = self.tournament_selection(population, fitnesses, selection_rate)
            # Create offspring using crossover and mutation
            offspring = self.crossover(parents, mutation_rate)
            # Replace worst individual with offspring
            population[population_size // 2:] = offspring
            # Update population with new individuals
            population.extend(self.mutation(population, mutation_rate))
        # Return best individual
        return population[0]

    def tournament_selection(self, population, fitnesses, selection_rate):
        # Select parents using tournament selection
        winners = random.choices(population, weights=fitnesses, k=population_size)
        # Select parents using selection rate
        winners = winners[:int(selection_rate * population_size)]
        return winners

    def crossover(self, parents, mutation_rate):
        # Create offspring using crossover
        offspring = []
        for _ in range(population_size // 2):
            parent1, parent2 = random.sample(parents, 2)
            child = (parent1 + parent2) / 2
            # Apply mutation
            if random.random() < mutation_rate:
                child = random.uniform(-1, 1)
            offspring.append(child)
        return offspring

    def mutation(self, population, mutation_rate):
        # Create new individuals with random mutations
        new_population = []
        for individual in population:
            new_individual = individual.copy()
            if random.random() < mutation_rate:
                new_individual[random.randint(0, self.dim - 1)] = random.uniform(-1, 1)
            new_population.append(new_individual)
        return new_population

# Description: Evolutionary Algorithm for Black Box Optimization
# Code: 