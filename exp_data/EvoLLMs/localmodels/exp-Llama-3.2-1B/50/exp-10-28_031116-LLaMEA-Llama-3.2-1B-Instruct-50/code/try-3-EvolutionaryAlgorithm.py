import numpy as np
import random

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = self.generate_population()
        self.fitnesses = self.calculate_fitnesses(self.population)
        self.best_solution = self.population[0]

    def generate_population(self):
        # Generate a population of candidate solutions
        population = []
        for _ in range(self.budget):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(solution)
        return population

    def calculate_fitnesses(self, population):
        # Calculate the fitness of each candidate solution
        fitnesses = []
        for solution in population:
            func = np.vectorize(lambda x: x**2)(solution)
            fitness = 1 / np.sum(np.abs(func))
            fitnesses.append(fitness)
        return fitnesses

    def __call__(self, func):
        # Optimize the black box function using the budget function evaluations
        for _ in range(self.budget):
            # Select the individual with the highest fitness
            best_individual = np.argmax(self.fitnesses)
            # Select a random individual
            random_individual = np.random.choice(self.population, p=self.fitnesses)
            # Perform a mutation on the best individual
            if random.random() < 0.45:
                mutation = random.uniform(-5.0, 5.0)
                best_individual[best_individual.index(min(best_individual))] += mutation
            # Perform a crossover on the best individual and the random individual
            if random.random() < 0.45:
                crossover = random.choice([0, 1])
                if crossover == 0:
                    best_individual = np.concatenate((best_individual, random_individual))
                else:
                    best_individual = np.concatenate((random_individual, best_individual))
            # Update the best solution
            self.best_solution = np.min(self.population, axis=0)
        return self.best_solution

# One-line description with the main idea
# Evolutionary Algorithm with Adaptive Mutation and Crossover

# Code