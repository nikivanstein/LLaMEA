import random
import numpy as np

class GeneticOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.population_size = 100
        self.mutation_rate = 0.01

    def __call__(self, func):
        # Initialize the population with random individuals
        population = self.generate_population(self.population_size, self.dim)

        # Evaluate the fitness of each individual
        fitnesses = [self.evaluate_fitness(individual, func) for individual in population]

        # Select the fittest individuals
        fittest_individuals = self.select_fittest(population, fitnesses)

        # Create a new generation by crossover and mutation
        new_population = self.create_new_generation(fittest_individuals, population, self.budget)

        # Replace the old population with the new one
        population = new_population

        # Update the function evaluations
        self.func_evaluations += len(new_population)

        # Return the best individual found so far
        return self.search_space[0], self.search_space[1]

    def generate_population(self, population_size, dim):
        return [np.random.uniform(self.search_space[0], self.search_space[1], (population_size, dim)) for _ in range(population_size)]

    def select_fittest(self, population, fitnesses):
        # Use tournament selection to select the fittest individuals
        tournament_size = 3
        winners = []
        for _ in range(len(population)):
            winner = random.choice(population)
            for _ in range(tournament_size):
                winner = random.choice(population)
                if winner[fitnesses.index(winner) < fitnesses.index(winner)]:
                    winner = winner
            winners.append(winner)
        return winners

    def create_new_generation(self, fittest_individuals, population, budget):
        new_population = []
        while len(new_population) < budget and len(fittest_individuals) > 0:
            # Select two parents using tournament selection
            parent1, parent2 = random.sample(fittest_individuals, 2)
            # Create a new individual by crossover and mutation
            child = self.crossover(parent1, parent2)
            # Apply mutation
            if random.random() < self.mutation_rate:
                child[random.randint(0, dim-1)] += np.random.uniform(-1, 1)
            new_population.append(child)
        return new_population

    def crossover(self, parent1, parent2):
        # Use single-point crossover
        index = random.randint(0, dim-1)
        child = parent1[:index] + parent2[index:]
        return child

# Description: Novel Metaheuristic Algorithm for Black Box Optimization using Genetic Algorithm
# Code: 