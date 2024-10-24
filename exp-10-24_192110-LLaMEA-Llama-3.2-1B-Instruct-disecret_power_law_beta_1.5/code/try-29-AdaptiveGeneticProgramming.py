import numpy as np
import random

class AdaptiveGeneticProgramming:
    def __init__(self, budget, dim, mutation_rate, alpha, beta, epsilon):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.population = []
        self.fitnesses = []

    def __call__(self, func):
        # Evaluate the function a specified number of times within the budget
        num_evals = min(self.budget, self.dim)
        for _ in range(num_evals):
            func_value = func()
        self.fitnesses.append(func_value)

        # Initialize the population with random solutions
        self.population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(100)]

        # Evolve the population using genetic programming
        while len(self.population) < 1000:
            # Select parents using tournament selection
            parents = self.select_parents()

            # Crossover (recombination) to create offspring
            offspring = self.crossover(parents)

            # Mutate the offspring to introduce genetic variation
            mutated_offspring = self.mutate(offspring)

            # Evaluate the fitness of each individual
            fitnesses = [self.evaluate(func, individual) for individual in mutated_offspring]

            # Select the fittest individuals to reproduce
            self.population = self.select_parents(fitnesses)

            # Update the population using the selected parents
            self.population = self.update_population(parents, fitnesses, self.alpha, self.beta)

        # Select the fittest individual to replace the oldest one
        self.population = self.select_fittest()

    def select_parents(self):
        # Select parents using tournament selection
        parents = []
        for _ in range(10):
            tournament_size = random.randint(1, 5)
            parents.append(self.select_tournament(self.population, tournament_size))
        return parents

    def crossover(self, parents):
        # Crossover (recombination) to create offspring
        offspring = []
        for _ in range(len(parents)):
            parent1, parent2 = random.sample(parents, 2)
            child = (parent1 + parent2) / 2
            offspring.append(child)
        return offspring

    def mutate(self, offspring):
        # Mutate the offspring to introduce genetic variation
        mutated_offspring = []
        for individual in offspring:
            mutated_individual = individual.copy()
            if random.random() < self.mutation_rate:
                mutated_individual[random.randint(0, self.dim - 1)] += random.uniform(-1, 1)
            mutated_offspring.append(mutated_individual)
        return mutated_offspring

    def evaluate(self, func, individual):
        # Evaluate the function at the given individual
        return func(individual)

    def select_fittest(self):
        # Select the fittest individual to replace the oldest one
        return self.population[0]

    def update_population(self, parents, fitnesses, alpha, beta):
        # Update the population using the selected parents
        new_population = []
        for _ in range(len(parents)):
            parent = parents.pop(0)
            fitness = fitnesses.pop(0)
            new_individual = self.evaluate(func, parent)
            new_population.append((new_individual, fitness))
            if len(new_population) >= len(parents):
                break
        return new_population