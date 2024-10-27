import numpy as np
import random

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.population_size = 100
        self.mutation_rate = 0.01
        self.evolution_strategy = "uniform"

    def __call__(self, func, budget):
        # Initialize population with random solutions
        population = [np.random.uniform(self.search_space) for _ in range(self.population_size)]

        # Evaluate function for each individual in the population
        fitnesses = [self.evaluate_fitness(individual, func, budget) for individual in population]

        # Select parents based on fitness
        parents = self.select_parents(fitnesses)

        # Create new offspring by crossover and mutation
        offspring = self.create_offspring(parents, population)

        # Evaluate function for each individual in the new population
        new_fitnesses = [self.evaluate_fitness(individual, func, budget) for individual in offspring]

        # Replace old population with new one
        population = offspring

        # If fitness of new population exceeds the budget, return it
        if np.max(new_fitnesses) > budget:
            return np.max(new_fitnesses)

        # Return the best individual in the new population
        return np.max(population)

    def select_parents(self, fitnesses):
        # Use roulette wheel selection with evolutionary strategy
        probabilities = []
        for i, fitness in enumerate(fitnesses):
            probability = 1 / (1 + (fitness / budget))
            probabilities.append(probability)
        probabilities = np.array(probabilities) / np.sum(probabilities)
        return np.random.choice(len(fitnesses), size=self.population_size, p=probabilities)

    def create_offspring(self, parents, population):
        # Use crossover and mutation with evolutionary strategy
        offspring = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(parents, 2)
            child = (parent1 + parent2) / 2
            if random.random() < self.mutation_rate:
                child += random.uniform(-1, 1)
            offspring.append(child)
        return offspring

    def evaluate_fitness(self, individual, func, budget):
        return func(individual)

# One-line description with the main idea
# Genetic Algorithm with Evolutionary Strategies
# 
# Uses a population of random solutions, selects parents based on fitness, creates new offspring by crossover and mutation, and evaluates the best individual in the new population.
# 
# Parameters:
#   budget (float): Maximum number of function evaluations
#   dim (int): Dimensionality of the search space
# 
# Returns:
#   The best individual in the new population, or the maximum fitness of the new population if it exceeds the budget.